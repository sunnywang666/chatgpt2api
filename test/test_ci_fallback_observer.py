import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/ci/provider-company-queue-fallback.sh"
WORKFLOW_SHA = "a" * 40
SOURCE_SHA = "b" * 40


FAKE_CURL = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

state_path = Path(os.environ["FAKE_STATE"])
log_path = Path(os.environ["FAKE_LOG"])
state = json.loads(state_path.read_text())
args = sys.argv[1:]
url = args[-1]
method = "POST" if "-X" in args and args[args.index("-X") + 1] == "POST" else "GET"
payload = args[args.index("-d") + 1] if "-d" in args else None
log_path.open("a").write(json.dumps({"method": method, "url": url, "payload": payload}) + "\n")

def queued_jobs():
    return {"total_count": 1, "jobs": [{"status": "queued", "conclusion": None, "runner_id": None,
      "runner_name": None, "steps": []}]}

def started_jobs():
    return {"total_count": 1, "jobs": [{"status": "in_progress", "conclusion": None, "runner_id": 9,
      "runner_name": "company", "steps": [{"status": "in_progress", "started_at": "now"}]}]}

def cancelled_jobs():
    return {"total_count": 1, "jobs": [{"status": "completed", "conclusion": "cancelled", "runner_id": None,
      "runner_name": None, "steps": []}]}

if method == "POST" and url.endswith("/cancel"):
    state["phase"] = "race" if state["scenario"] == "race" else "cancelled"
    state_path.write_text(json.dumps(state))
    raise SystemExit(0)
if method == "POST" and url.endswith("/dispatches"):
    raise SystemExit(0)
if "/actions/runs/42/jobs" in url:
    value = started_jobs() if state["scenario"] == "started" or state["phase"] == "race" else (cancelled_jobs() if state["phase"] == "cancelled" else queued_jobs())
    if state["scenario"] == "partial-jobs":
        value["total_count"] = 2
elif "/actions/runs/42" in url:
    value = {"event": os.environ["FAKE_EVENT"], "display_title": os.environ["FAKE_REST_DISPLAY_TITLE"], "head_sha": os.environ["WORKFLOW_SHA"],
      "run_attempt": 2 if state["phase"] == "attempt-changed" else 1,
      "status": "in_progress", "conclusion": None}
    if state["phase"] in {"cancelled", "race"}:
        value.update(status="completed", conclusion="cancelled")
elif "/git/ref/heads/" in url:
    value = {"object": {"sha": os.environ["WORKFLOW_SHA"]}}
elif "/compare/" in url:
    value = {"status": "ahead"}
elif "/contents/" in url:
    if state["scenario"] == "blob-start":
        state["phase"] = "race"
        state_path.write_text(json.dumps(state))
    elif state["scenario"] == "attempt-change":
        state["phase"] = "attempt-changed"
        state_path.write_text(json.dumps(state))
    value = {"sha": "c" * 40}
else:
    raise SystemExit("unexpected URL " + url)
print(json.dumps(value))
'''


def run_observer(scenario: str, workflow_name: str, title: str, event: str = "push", rest_title: str | None = None):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        fake_curl = root / "curl"
        fake_curl.write_text(FAKE_CURL)
        fake_curl.chmod(0o755)
        fake_sleep = root / "sleep"
        fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_sleep.chmod(0o755)
        state = root / "state.json"
        state.write_text(json.dumps({"scenario": scenario, "phase": "before"}))
        log = root / "log.jsonl"
        env = os.environ | {
            "GH_TOKEN": "test-token",
            "REPOSITORY": "sunnywang666/chatgpt2api",
            "RUN_ID": "42",
            "SOURCE_ATTEMPT": "1",
            "WORKFLOW_NAME": workflow_name,
            "WORKFLOW_SHA": WORKFLOW_SHA,
            "SOURCE_BRANCH": "main",
            "DISPLAY_TITLE": title,
            "FAKE_REST_DISPLAY_TITLE": rest_title or title,
            "FAKE_EVENT": event,
            "RUNNER_TEMP": str(root),
            "CURL_BIN": str(fake_curl),
            "OBSERVE_MAX_SECONDS": "0",
            "PATH": f"{root}:{os.environ['PATH']}",
            "FAKE_STATE": str(state),
            "FAKE_LOG": str(log),
        }
        result = subprocess.run(["bash", str(HELPER)], env=env, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls


def test_company_start_exits_without_cancel_or_dispatch():
    result, calls = run_observer("started", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode == 0, result.stderr
    assert not any(call["method"] == "POST" for call in calls)


def test_queued_company_run_cancels_then_dispatches_hosted_exact_sha():
    result, calls = run_observer("queued", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode == 0, result.stderr
    post_calls = [call for call in calls if call["method"] == "POST"]
    assert post_calls[0]["url"].endswith("/actions/runs/42/cancel")
    payload = json.loads(post_calls[1]["payload"])
    assert payload == {"ref": "main", "inputs": {"source_sha": WORKFLOW_SHA, "github_hosted": True}}


def test_cancellation_race_is_reported_without_dispatch():
    result, calls = run_observer("race", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode != 0
    assert "started during cancellation" in result.stdout
    assert sum(call["method"] == "POST" for call in calls) == 1


def test_publisher_uses_run_name_source_ref_not_workflow_sha():
    result, calls = run_observer("queued", "Publish Docker Image", f"Publish Docker Image {SOURCE_SHA} company", event="workflow_dispatch")

    assert result.returncode == 0, result.stderr
    dispatch = [call for call in calls if call["url"].endswith("/dispatches")][0]
    payload = json.loads(dispatch["payload"])
    assert payload == {"ref": "main", "inputs": {"source_ref": SOURCE_SHA, "github_hosted": True}}


def test_publisher_uses_fresh_rest_title_when_requested_event_is_stale():
    result, calls = run_observer(
        "queued",
        "Publish Docker Image",
        "Publish Docker Image",
        event="workflow_dispatch",
        rest_title=f"Publish Docker Image {SOURCE_SHA} company",
    )

    assert result.returncode == 0, result.stderr
    dispatch = [call for call in calls if call["url"].endswith("/dispatches")][0]
    assert json.loads(dispatch["payload"]) == {
        "ref": "main",
        "inputs": {"source_ref": SOURCE_SHA, "github_hosted": True},
    }


def test_start_during_workflow_blob_reads_does_not_cancel():
    result, calls = run_observer("blob-start", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode == 0, result.stderr
    assert not any(call["method"] == "POST" for call in calls)


def test_rerun_attempt_change_does_not_cancel():
    result, calls = run_observer("attempt-change", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode == 0, result.stderr
    assert not any(call["method"] == "POST" for call in calls)


def test_partial_jobs_response_is_not_cancelled():
    result, calls = run_observer("partial-jobs", "Provider POOL-02 isolated review", f"Provider POOL-02 {WORKFLOW_SHA} company")

    assert result.returncode == 0, result.stderr
    assert not any(call["method"] == "POST" for call in calls)


def test_tag_publisher_run_is_never_cancelled_or_redispatched():
    result, calls = run_observer("queued", "Publish Docker Image", f"Publish Docker Image {SOURCE_SHA} company", event="push")

    assert result.returncode == 0, result.stderr
    assert not any(call["method"] == "POST" for call in calls)
