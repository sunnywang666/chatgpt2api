from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pool_ci_dispatch_is_exact_source_and_company_by_default():
    workflow = (ROOT / ".github/workflows/pool02-scoped-ci.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "source_sha:" in workflow
    assert "inputs.github_hosted == true && 'ubuntu-latest'" in workflow
    assert 'fromJSON(\'["self-hosted","Linux","X64","hugsweet-linux-ci"]\')' in workflow
    assert 'test "$DISPATCH_SHA" = "$SOURCE_SHA"' in workflow
    assert 'ref: ${{ env.SOURCE_SHA }}' in workflow


def test_queue_observer_can_only_replay_untouched_runs():
    workflow = (ROOT / ".github/workflows/company-queue-fallback.yml").read_text()

    assert "workflow_run:" in workflow
    assert "requested" in workflow
    assert "actions: write" in workflow
    assert "actions/checkout@v5" in workflow
    assert "bash tools/ci/provider-company-queue-fallback.sh" in workflow
    assert "auto-hosted" in workflow
    assert "secrets." not in workflow
    assert "vars." not in workflow

    helper = (ROOT / "tools/ci/provider-company-queue-fallback.sh").read_text()
    assert 'OBSERVE_MAX_SECONDS="${OBSERVE_MAX_SECONDS:-90}"' in helper
    assert '(.runner_id != null) and (.runner_id != 0)' in helper
    assert '.run_attempt == ($attempt | tonumber)' in helper
    assert '.total_count == (.jobs | length)' in helper
    assert 'SOURCE_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}' in workflow
    assert 'The branch and blob reads above can take time.' in helper
    assert 'actions/runs/$RUN_ID/cancel' in helper
    assert "is_untouched true" in helper
    assert "Company job started during cancellation; refusing hosted fallback." in helper
    assert "source_blob" in helper and "dispatch_blob" in helper
    assert 'Publish\\ Docker\\ Image\\ ([a-f0-9]{40})\\ company' in helper
    assert "Tag-push releases retain version/latest and multi-architecture semantics" in helper
    assert "jq -e '.event == \"workflow_dispatch\"'" in helper
    assert "inputs: $inputs" in helper


def test_docker_publisher_marks_automatic_hosted_replays():
    workflow = (ROOT / ".github/workflows/docker-publish.yml").read_text()

    assert "source_ref:" in workflow
    assert "github_hosted:" in workflow
    assert "auto-hosted" in workflow
