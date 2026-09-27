"""
Publish the pipeline code to the Databricks workspace and create (or update)
the Databricks Job that runs it on serverless compute:

    bronze -> silver -> silver_label -> gold -> parity

Idempotent: re-running uploads the current code and resets the job's
definition in place (same job id), so it never duplicates anything.

    python databricks/deploy_job.py            # upload code + create/update job
    python databricks/deploy_job.py --run      # ... and trigger a run, wait for it
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pipeline import _load_dotenv  # noqa: E402
from pipeline.common import REPO_ROOT  # noqa: E402

JOB_NAME = "claims-denial-medallion-pipeline"
STEPS = ["bronze", "silver", "silver_label", "gold", "parity"]
CODE_GLOBS = ["src/*.py", "databricks/*.py", "databricks/pipeline/*.py", "deploy/score.py"]


def upload_code(w, root: str) -> None:
    from databricks.sdk.service.workspace import ImportFormat

    for pattern in CODE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            rel = path.relative_to(REPO_ROOT).as_posix()
            target = f"{root}/{rel}"
            w.workspace.mkdirs(target.rsplit("/", 1)[0])
            with open(path, "rb") as fh:
                w.workspace.upload(target, fh, format=ImportFormat.AUTO, overwrite=True)
    print(f"  code uploaded to {root}")


def job_settings(root: str):
    from databricks.sdk.service import compute, jobs

    tasks = []
    for i, step in enumerate(STEPS):
        tasks.append(jobs.Task(
            task_key=step,
            depends_on=[jobs.TaskDependency(task_key=STEPS[i - 1])] if i else None,
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"{root}/databricks/run_pipeline.py",
                parameters=["--steps", step, "--repo-root", root]),
            environment_key="pipeline",
        ))
    return dict(
        name=JOB_NAME,
        description="CMS claims denial: raw CSV (bronze) -> typed claims + engineered label (silver) "
                    "-> model-ready features (gold), then a row-level parity check against the pandas pipeline.",
        tasks=tasks,
        environments=[jobs.JobEnvironment(
            environment_key="pipeline",
            spec=compute.Environment(environment_version="3", dependencies=["scikit-learn", "statsmodels"]))],
        max_concurrent_runs=1,
        tags={"project": "claims-denial-risk-prediction"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    _load_dotenv()

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import jobs

    w = WorkspaceClient()
    # (the bare project name is already taken by the MLflow experiment)
    root = f"/Workspace/Users/{w.current_user.me().user_name}/claims-denial-pipeline"
    upload_code(w, root)

    settings = job_settings(root)
    existing = [j for j in w.jobs.list(name=JOB_NAME)]
    if existing:
        job_id = existing[0].job_id
        w.jobs.reset(job_id=job_id, new_settings=jobs.JobSettings(**settings))
        print(f"  updated job {JOB_NAME} ({job_id})")
    else:
        job_id = w.jobs.create(**settings).job_id
        print(f"  created job {JOB_NAME} ({job_id})")

    if args.run:
        run = w.jobs.run_now(job_id=job_id)
        print(f"  run {run.run_id} started; waiting ...")
        while True:
            r = w.jobs.get_run(run.run_id)
            state = r.state.life_cycle_state.value
            if state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
                break
            time.sleep(30)
        for t in r.tasks or []:
            print(f"    {t.task_key:13s} {t.state.result_state.value if t.state.result_state else t.state.life_cycle_state.value}")
        print(f"  result: {r.state.result_state.value if r.state.result_state else state}  {r.run_page_url}")


if __name__ == "__main__":
    main()
