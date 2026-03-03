"""Collect a fixed cohort of public GitHub Actions job metadata."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


@dataclass(frozen=True)
class Cohort:
    key: str
    title: str
    repo: str
    workflow: int
    created: str
    output: Path


PANDAS = Cohort("pandas", "pandas Unit Tests", "pandas-dev/pandas", 54497638,
                "2026-11-21..2026-11-23", Path("results/raw/ci-jobs.json"))
SCIPY = Cohort("scipy", "SciPy Linux tests", "scipy/scipy", 3941552,
               "2026-11-17..2026-11-23", Path("results/raw/ci-jobs-scipy.json"))
COHORTS = {cohort.key: cohort for cohort in (PANDAS, SCIPY)}


def gh(endpoint: str) -> dict:
    response = subprocess.run(["gh", "api", endpoint], capture_output=True, check=True, encoding="utf-8", timeout=45)
    return json.loads(response.stdout)


def list_records(endpoint: str, key: str, **params) -> list[dict]:
    query = urlencode({**params, "per_page": 100})
    url = f"{endpoint}?{query}"
    first = gh(url)
    total = first["total_count"]
    rows = list(first[key])
    for page in range(2, (total + 99) // 100 + 1):
        result = gh(f"{url}&page={page}")
        if result["total_count"] != total:
            raise ValueError(f"Count changed while paging {key}")
        rows.extend(result[key])
    if len(rows) != total or len({int(row["id"]) for row in rows}) != total:
        raise ValueError(f"Missing or duplicate records while paging {key}")
    return rows


def collect(cohort: Cohort = PANDAS) -> dict:
    runs = list_records(f"repos/{cohort.repo}/actions/workflows/{cohort.workflow}/runs", "workflow_runs",
                        event="pull_request", status="completed", created=cohort.created)
    if not runs:
        raise ValueError("No completed runs in the pinned cohort")

    def jobs(run: dict) -> list[dict]:
        run_id = int(run["id"])
        records = list_records(f"repos/{cohort.repo}/actions/runs/{run_id}/attempts/1/jobs", "jobs")
        keys = ("id", "name", "started_at", "completed_at", "conclusion")
        return [dict(run_id=run_id, **{key: job.get(key) for key in keys}) for job in records]

    with ThreadPoolExecutor(max_workers=5) as executor:
        rows = [row for run_jobs in executor.map(jobs, runs) for row in run_jobs]
    return {"repo": cohort.repo, "workflow": cohort.workflow, "created_window": cohort.created, "attempt": 1,
            "collected_at": datetime.now(timezone.utc).isoformat(), "runs": len(runs), "jobs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, default="pandas")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    cohort = COHORTS[args.cohort]
    out = args.out or cohort.output
    data = collect(cohort)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(data['jobs'])} jobs across {data['runs']} workflow runs: {out.resolve()}")


if __name__ == "__main__":
    main()
