#!/usr/bin/env bash
# Build a local .env from .env.example by prompting for each value.
# Secrets (names containing KEY, TOKEN or SECRET) are typed hidden, so they never
# appear on screen, in shell history, or in chat. Press Enter to keep the default.
# Usage (Git Bash, repo root):  bash scripts/make_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env ]]; then
  echo ".env already exists. Edit it directly, or move it aside first."
  exit 1
fi

umask 077
: > .env
while IFS= read -r line <&3 || [[ -n "$line" ]]; do
  line="${line%$'\r'}"
  if [[ -z "$line" || "$line" == \#* ]]; then
    continue
  fi
  key="${line%%=*}"
  default="${line#*=}"
  if [[ "$key" =~ (KEY|TOKEN|SECRET) ]]; then
    read -r -s -p "$key (hidden, Enter to leave blank): " val; echo
  else
    read -r -p "$key [$default]: " val
    val="${val:-$default}"
  fi
  printf '%s=%s\n' "$key" "$val" >> .env
done 3< .env.example

echo "Wrote .env ($(wc -l < .env) variables). It is gitignored; do not commit it."
