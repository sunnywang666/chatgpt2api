#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?} ${REPOSITORY:?} ${RUN_ID:?} ${SOURCE_ATTEMPT:?} ${WORKFLOW_NAME:?} ${WORKFLOW_SHA:?} ${SOURCE_BRANCH:?} ${DISPLAY_TITLE:?}"
CURL_BIN="${CURL_BIN:-curl}"
OBSERVE_MAX_SECONDS="${OBSERVE_MAX_SECONDS:-90}"
OBSERVE_INTERVAL_SECONDS="${OBSERVE_INTERVAL_SECONDS:-5}"
STATE_DIR="${RUNNER_TEMP:-$(mktemp -d)}"
RUN_JSON="$STATE_DIR/source-run.json"
JOBS_JSON="$STATE_DIR/source-jobs.json"

api() {
  "$CURL_BIN" --fail --silent --show-error --connect-timeout 15 --max-time 60 \
    -H "Authorization: Bearer $GH_TOKEN" -H 'Accept: application/vnd.github+json' \
    "https://api.github.com$1"
}

post() {
  "$CURL_BIN" --fail --silent --show-error --connect-timeout 15 --max-time 60 \
    -X POST -H "Authorization: Bearer $GH_TOKEN" -H 'Accept: application/vnd.github+json' "$@"
}

[[ "$RUN_ID" =~ ^[0-9]+$ && "$SOURCE_ATTEMPT" =~ ^[0-9]+$ && "$WORKFLOW_SHA" =~ ^[a-f0-9]{40}$ ]]

fetch_source() {
  api "/repos/$REPOSITORY/actions/runs/$RUN_ID" > "$RUN_JSON"
  api "/repos/$REPOSITORY/actions/runs/$RUN_ID/jobs?filter=latest&per_page=100" > "$JOBS_JSON"
}

has_started() {
  jq -e '.status == "completed" or .conclusion != null' "$RUN_JSON" >/dev/null && return 0
  jq -e 'any(.jobs[]?;
    .status == "in_progress" or .status == "completed" or
    ((.runner_id != null) and (.runner_id != 0)) or
    ((.runner_name // "") != "") or
    any(.steps[]?; .status == "in_progress" or .status == "completed" or .started_at != null)
  )' "$JOBS_JSON" >/dev/null
}

is_untouched() {
  local allow_cancelled="${1:-false}"
  jq -e --arg sha "$WORKFLOW_SHA" --arg attempt "$SOURCE_ATTEMPT" --arg allow_cancelled "$allow_cancelled" '
    (.event == "push" or .event == "workflow_dispatch") and .head_sha == $sha and
    .run_attempt == ($attempt | tonumber) and
    (if $allow_cancelled == "true" then .status == "completed" and .conclusion == "cancelled"
     else (.status == "queued" or .status == "in_progress") and .conclusion == null end)
  ' "$RUN_JSON" >/dev/null || return 1
  jq -e --arg allow_cancelled "$allow_cancelled" '
    (.total_count | type == "number") and .total_count == (.jobs | length) and
    (.jobs | length) > 0 and all(.jobs[];
      (if $allow_cancelled == "true"
       then (.status == "queued" and .conclusion == null) or (.status == "completed" and .conclusion == "cancelled")
       else .status == "queued" and .conclusion == null end) and
      (.runner_id == null or .runner_id == 0) and (.runner_name == null or .runner_name == "") and
      all(.steps[]?; .started_at == null and
        (if $allow_cancelled == "true" then .status == "queued" or (.status == "completed" and .conclusion == "cancelled")
         else .status == "queued" and .completed_at == null end))
    )
  ' "$JOBS_JSON" >/dev/null
}

# Leave healthy company work immediately. If jobs have not appeared, keep
# observing until the 90-second upper bound; API failures end this script.
for ((elapsed = 0; elapsed < OBSERVE_MAX_SECONDS; elapsed += OBSERVE_INTERVAL_SECONDS)); do
  fetch_source
  if has_started; then
    echo 'Company run started or finished; no fallback.'
    exit 0
  fi
  sleep "$OBSERVE_INTERVAL_SECONDS"
done
fetch_source
if has_started || ! is_untouched; then
  echo 'Company run started, finished, changed attempt, or is not provably untouched; no fallback.'
  exit 0
fi

SOURCE_SHA="$WORKFLOW_SHA"
case "$WORKFLOW_NAME" in
  'Provider POOL-02 isolated review')
    [[ "$SOURCE_BRANCH" =~ ^(main|codex/[A-Za-z0-9._/-]+)$ ]] || exit 0
    branch_ref="$(jq -rn --arg branch "$SOURCE_BRANCH" '$branch | @uri')"
    test "$(api "/repos/$REPOSITORY/git/ref/heads/$branch_ref" | jq -r '.object.sha')" = "$WORKFLOW_SHA" || exit 0
    workflow_file='pool02-scoped-ci.yml'
    dispatch_ref="$SOURCE_BRANCH"
    dispatch_inputs="$(jq -nc --arg source_sha "$SOURCE_SHA" '{source_sha: $source_sha, github_hosted: true}')"
    ;;
  'Publish Docker Image')
    # Tag-push releases retain version/latest and multi-architecture semantics;
    # only the existing workflow_dispatch contract can be replayed safely.
    jq -e '.event == "workflow_dispatch"' "$RUN_JSON" >/dev/null || exit 0
    [[ "$DISPLAY_TITLE" =~ ^Publish\ Docker\ Image\ ([a-f0-9]{40})\ company$ ]] || exit 0
    SOURCE_SHA="${BASH_REMATCH[1]}"
    workflow_file='docker-publish.yml'
    dispatch_ref='main'
    main_sha="$(api "/repos/$REPOSITORY/git/ref/heads/main" | jq -r '.object.sha')"
    [[ "$main_sha" =~ ^[a-f0-9]{40}$ ]]
    case "$(api "/repos/$REPOSITORY/compare/$SOURCE_SHA...$main_sha" | jq -r '.status')" in ahead|identical) ;; *) exit 0 ;; esac
    dispatch_inputs="$(jq -nc --arg source_ref "$SOURCE_SHA" '{source_ref: $source_ref, github_hosted: true}')"
    ;;
  *) exit 0 ;;
esac

source_blob="$(api "/repos/$REPOSITORY/contents/.github/workflows/$workflow_file?ref=$WORKFLOW_SHA" | jq -r '.sha')"
dispatch_blob="$(api "/repos/$REPOSITORY/contents/.github/workflows/$workflow_file?ref=$dispatch_ref" | jq -r '.sha')"
[[ "$source_blob" =~ ^[a-f0-9]{40}$ && "$source_blob" == "$dispatch_blob" ]] || exit 0

# The branch and blob reads above can take time. Re-read immediately before
# cancellation so an active company attempt is never deliberately interrupted.
fetch_source
if has_started || ! is_untouched; then
  echo 'Company run started, finished, changed attempt, or is not provably untouched; no fallback.'
  exit 0
fi
post "https://api.github.com/repos/$REPOSITORY/actions/runs/$RUN_ID/cancel" >/dev/null
for _ in {1..30}; do
  sleep 1
  fetch_source
  if [[ "$(jq -r '.status' "$RUN_JSON")" == completed ]]; then
    test "$(jq -r '.conclusion' "$RUN_JSON")" = cancelled || exit 1
    break
  fi
done
test "$(jq -r '.status' "$RUN_JSON")" = completed
if ! is_untouched true; then
  echo '::error::Company job started during cancellation; refusing hosted fallback.'
  exit 1
fi

payload="$(jq -nc --arg ref "$dispatch_ref" --argjson inputs "$dispatch_inputs" '{ref: $ref, inputs: $inputs}')"
post -d "$payload" "https://api.github.com/repos/$REPOSITORY/actions/workflows/$workflow_file/dispatches" >/dev/null
echo 'Hosted fallback dispatched after confirmed untouched cancellation.'
