#!/usr/bin/env bash
# Phase 3 -- stand up (or update) the Azure ML deployment of the Phase 2
# XGBoost claims-denial model. Idempotent: every step checks what already
# exists and skips it, so re-running this from a new session, or from your
# own machine, never duplicates anything.
#
# Prereqs (one time per machine):
#   - Azure CLI + its ml extension:  az extension add -n ml
#   - az login --tenant "$AZURE_TENANT_ID"
#   - .env at the repo root with AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID,
#     AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE, AZURE_LOCATION
#   - reports/xgboost_model.json + reports/xgboost_model_meta.json present,
#     ONLY for the first registration (`python src/fit_xgboost.py`, or
#     `deploy/deploy.sh download-model` once it's registered)
#
# Usage (repo root, Git Bash or Linux):
#   deploy/deploy.sh               # everything: rg, workspace, model, endpoint, deployment, smoke test
#   deploy/deploy.sh test          # score deploy/sample_request.json against the live endpoint
#   deploy/deploy.sh download-model  # pull the registered model files into reports/
#   deploy/deploy.sh status        # endpoint + deployment state
#   deploy/deploy.sh stop          # delete the deployment (stops VM billing; endpoint/model kept)
#   deploy/deploy.sh teardown      # delete the endpoint entirely (model + workspace kept)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then echo "missing .env at repo root (copy .env.example)"; exit 1; fi
set -a; source <(sed 's/\r$//' .env); set +a
: "${AZURE_SUBSCRIPTION_ID:?}" "${AZURE_RESOURCE_GROUP:?}" "${AZURE_ML_WORKSPACE:?}" "${AZURE_LOCATION:?}"

AZ="${AZ:-az}"
MODEL_NAME="claims-denial-xgboost"
ENDPOINT_NAME="claims-denial-xgb"
DEPLOYMENT_NAME="blue"
WS=(--resource-group "$AZURE_RESOURCE_GROUP" --workspace-name "$AZURE_ML_WORKSPACE")

"$AZ" account set --subscription "$AZURE_SUBSCRIPTION_ID"

model_hash() {  # content fingerprint of the local model files
  cat reports/xgboost_model.json reports/xgboost_model_meta.json | sha256sum | cut -c1-16
}

ensure_infra() {
  if [[ "$("$AZ" group exists -n "$AZURE_RESOURCE_GROUP")" != "true" ]]; then
    echo "== creating resource group $AZURE_RESOURCE_GROUP ($AZURE_LOCATION)"
    "$AZ" group create -n "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" --tags project=claims-denial-risk-prediction -o none
  fi
  if ! "$AZ" ml workspace show -n "$AZURE_ML_WORKSPACE" -g "$AZURE_RESOURCE_GROUP" -o none 2>/dev/null; then
    echo "== creating workspace $AZURE_ML_WORKSPACE (takes a few minutes)"
    "$AZ" ml workspace create -n "$AZURE_ML_WORKSPACE" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" \
      --tags project=claims-denial-risk-prediction -o none
  fi
}

ensure_model() {
  local existing
  existing="$("$AZ" ml model list --name "$MODEL_NAME" "${WS[@]}" --query "[].tags.content_sha256_16" -o tsv 2>/dev/null || true)"
  if [[ ! -f reports/xgboost_model.json || ! -f reports/xgboost_model_meta.json ]]; then
    if [[ -n "$existing" ]]; then echo "== model already registered; no local copy needed"; return; fi
    echo "no registered model and no local reports/xgboost_model*.json -- run: python src/fit_xgboost.py"; exit 1
  fi
  local h; h="$(model_hash)"
  if grep -qx "$h" <<<"$existing"; then
    echo "== model $MODEL_NAME with content hash $h already registered"
    return
  fi
  echo "== registering $MODEL_NAME (content hash $h)"
  local stage; stage="$(mktemp -d)"
  mkdir -p "$stage/$MODEL_NAME"
  cp reports/xgboost_model.json reports/xgboost_model_meta.json "$stage/$MODEL_NAME/"
  "$AZ" ml model create --name "$MODEL_NAME" --type custom_model --path "$stage/$MODEL_NAME" "${WS[@]}" \
    --description "Phase 2 XGBoost (early-stopped, 89 trees). val PR-AUC 0.583 / ROC-AUC 0.816. Regenerable: python src/fit_xgboost.py" \
    --tags content_sha256_16="$h" val_pr_auc=0.5832 val_roc_auc=0.8161 git_repo=mohassan99/claims-denial-risk-prediction \
    -o none
  rm -rf "$stage"
}

ensure_endpoint() {
  if ! "$AZ" ml online-endpoint show -n "$ENDPOINT_NAME" "${WS[@]}" -o none 2>/dev/null; then
    echo "== creating endpoint $ENDPOINT_NAME"
    "$AZ" ml online-endpoint create -f deploy/endpoint.yml "${WS[@]}" -o none
    # The endpoint's managed identity gets AcrPull/Storage Blob Data Reader
    # automatically at create time, but role assignments take a minute or two
    # to propagate -- a deployment started immediately can fail to pull its
    # image. (First run, 2026-09-25: see SESSION_LOG.md.)
    echo "   waiting 120s for the endpoint identity's role assignments to propagate"
    sleep 120
  fi
}

ensure_deployment() {
  if "$AZ" ml online-deployment show -n "$DEPLOYMENT_NAME" -e "$ENDPOINT_NAME" "${WS[@]}" -o none 2>/dev/null; then
    echo "== updating deployment $DEPLOYMENT_NAME (picks up score.py/conda.yml/model changes; ~10 min)"
    "$AZ" ml online-deployment update -f deploy/deployment.yml "${WS[@]}" -o none
  else
    echo "== creating deployment $DEPLOYMENT_NAME (builds the image first; 10-20 min)"
    "$AZ" ml online-deployment create -f deploy/deployment.yml "${WS[@]}" --all-traffic -o none
  fi
  # `update` never touches traffic, and a create that failed part-way leaves
  # traffic unset -- so always (re)assert 100% to this deployment.
  "$AZ" ml online-endpoint update -n "$ENDPOINT_NAME" "${WS[@]}" --traffic "$DEPLOYMENT_NAME=100" -o none
}

smoke_test() {
  echo "== scoring deploy/sample_request.json"
  "$AZ" ml online-endpoint invoke -n "$ENDPOINT_NAME" "${WS[@]}" --request-file deploy/sample_request.json
  echo
}

case "${1:-all}" in
  all)            ensure_infra; ensure_model; ensure_endpoint; ensure_deployment; smoke_test ;;
  test)           smoke_test ;;
  download-model)
    tmp="$(mktemp -d)"
    ver="$("$AZ" ml model list --name "$MODEL_NAME" "${WS[@]}" --query "[0].version" -o tsv)"
    "$AZ" ml model download --name "$MODEL_NAME" --version "$ver" "${WS[@]}" --download-path "$tmp"
    find "$tmp" -name 'xgboost_model*.json' -exec cp {} reports/ \;
    echo "downloaded version $ver -> reports/ (content hash $(model_hash))" ;;
  status)
    "$AZ" ml online-endpoint show -n "$ENDPOINT_NAME" "${WS[@]}" --query "{state:provisioning_state,uri:scoring_uri,traffic:traffic}" -o json
    "$AZ" ml online-deployment list -e "$ENDPOINT_NAME" "${WS[@]}" --query "[].{name:name,state:provisioning_state,sku:instance_type,count:instance_count}" -o table ;;
  stop)           "$AZ" ml online-deployment delete -n "$DEPLOYMENT_NAME" -e "$ENDPOINT_NAME" "${WS[@]}" --yes ;;
  teardown)       "$AZ" ml online-endpoint delete -n "$ENDPOINT_NAME" "${WS[@]}" --yes ;;
  *) echo "unknown command: $1"; exit 1 ;;
esac
