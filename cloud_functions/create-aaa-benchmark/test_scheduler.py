# 2025-09-05

from datetime import datetime
from urllib.parse import urlparse
from pytz import timezone as tz
import pytz
from croniter import croniter
import pytest

from google.cloud import scheduler_v1
from google.cloud import functions_v2
from google.cloud import run_v2

# ---------- Hard-coded config ----------
PROJECT_ID = "eng-reactor-287421"
LOCATION = "us-east4"
FUNCTION_NAME = "create-aaa-benchmark"   # Cloud Function / Cloud Run service name
SCHEDULER_LOCATION = LOCATION


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def cfg():
    return {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "scheduler_location": SCHEDULER_LOCATION,
        "function_name": FUNCTION_NAME,
    }


@pytest.fixture(scope="module")
def scheduler_client():
    print("[fixture] Creating Cloud Scheduler client")
    return scheduler_v1.CloudSchedulerClient()


# ---------- Helpers ----------

def _human_next_run(cron_expr: str, time_zone: str):
    """Compute next run time for display only."""
    try:
        job_tz = tz(time_zone)
    except Exception:
        job_tz = pytz.UTC
    now = datetime.now(job_tz)
    it = croniter(cron_expr, now)
    nxt = it.get_next(datetime)
    return nxt.astimezone(job_tz)


def _find_function_uri(project_id: str, location: str, function_name: str) -> str | None:
    client = functions_v2.FunctionServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    print(f"[lookup] Searching Cloud Functions v2 under {parent} for '{function_name}'")

    for fn in client.list_functions(request={"parent": parent}):
        short_name = fn.name.split("/")[-1]
        if short_name == function_name:
            uri = (fn.service_config.uri or "").strip() if fn.service_config else ""
            print(f"[lookup] Found Cloud Function '{short_name}' with uri={uri}")
            return uri or None
    print("[lookup] No matching Cloud Function v2 found.")
    return None


def _find_run_service_uri(project_id: str, location: str, service_name: str) -> str | None:
    client = run_v2.ServicesClient()
    parent = f"projects/{project_id}/locations/{location}"
    print(f"[lookup] Searching Cloud Run services under {parent} for '{service_name}'")

    for svc in client.list_services(request={"parent": parent}):
        short_name = svc.name.split("/")[-1]
        if short_name == service_name:
            uri = (svc.uri or "").strip()
            print(f"[lookup] Found Cloud Run service '{short_name}' with uri={uri}")
            return uri or None
    print("[lookup] No matching Cloud Run service found.")
    return None


def _job_targets_service(job_uri: str, service_name: str, location: str) -> bool:
    """
    Accept both Cloud Run host styles:
      1) <service>-<hash>-<region>.a.run.app
      2) <service>-<project-number>.<region>.run.app
    Require host starts with '<service>-' AND ends with either:
      - '.a.run.app'  (global)
      - f'.{location}.run.app' (regional)
    """
    try:
        host = urlparse(job_uri).hostname or ""
    except Exception:
        return False

    if not host.startswith(f"{service_name}-"):
        return False

    if host.endswith(".a.run.app"):
        return True

    if host.endswith(f".{location}.run.app"):
        return True

    return False


# ---------- Test ----------

def test_function_and_scheduler_linkage(cfg, scheduler_client):
    project_id = cfg["project_id"]
    location = cfg["location"]
    scheduler_location = cfg["scheduler_location"]
    function_name = cfg["function_name"]

    # 1) Resolve a public URI
    uri = _find_function_uri(project_id, location, function_name)
    if not uri:
        uri = _find_run_service_uri(project_id, location, function_name)

    if not uri:
        pytest.fail(
            f"No deployed Cloud Function v2 or Cloud Run service named '{function_name}' "
            f"found in {project_id}/{location}. Cannot correlate Scheduler jobs."
        )

    print(f"[result] Using deployed URI: {uri}")

    # 2) List Scheduler jobs and match by hostname pattern
    parent = f"projects/{project_id}/locations/{scheduler_location}"
    jobs = list(scheduler_client.list_jobs(request={"parent": parent}))
    print(f"[scheduler] Found {len(jobs)} job(s) under {parent}")

    matched = 0
    paused = []

    for job in jobs:
        if not job.http_target or not job.http_target.uri:
            continue

        job_uri = job.http_target.uri.strip()
        if not _job_targets_service(job_uri, function_name, location):
            continue

        name = job.name.split("/")[-1]
        state = scheduler_v1.Job.State(job.state).name if job.state is not None else "UNKNOWN"
        schedule = job.schedule or "(no schedule)"
        time_zone = job.time_zone or "UTC"

        try:
            next_run = _human_next_run(schedule, time_zone) if schedule != "(no schedule)" else None
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z") if next_run else "n/a"
        except Exception as e:
            next_run_str = f"(failed to compute next run: {e})"

        print(
            f"[scheduler] Scheduler job for this function/service:\n"
            f"  name:     '{name}'\n"
            f"  status:   {state}\n"
            f"  schedule: '{schedule}'  tz={time_zone}\n"
            f"  next run: {next_run_str}\n"
            f"  target:   {job_uri}\n"
        )

        matched += 1
        if state != "ENABLED":
            paused.append((name, state))

    # Fail if none matched
    if matched == 0:
        pytest.fail("No Cloud Scheduler jobs matched this function/service.")

    # Fail if any matched job is not enabled
    if paused:
        details = ", ".join([f"{n} (state={s})" for n, s in paused])
        pytest.fail(f"One or more Scheduler jobs are not active: {details}")

    print(f"[result] {matched} matching Scheduler job(s) found and all are ENABLED.")
