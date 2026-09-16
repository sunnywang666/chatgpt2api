#!/usr/bin/env bash
# Run a real Codex CLI acceptance turn without loading the user's normal Codex state.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
base_url="https://app.hugsweetglobal.com/ai/codex/v1"
state_root="$script_dir/.codex-client-state"
fixture_root="$script_dir/acceptance-fixture"
model=""
resume_id=""
list_only=0
prompt="Fix only calculator.py so test_calculator.py passes. Run python -m unittest -v and report the result."

usage() {
  printf '%s\n' "Usage: CODEX_PROVIDER_API_KEY=... $0 --model MODEL [--resume SESSION_ID] [--list-models] [--base-url URL] [--state-root PATH] [--fixture-root PATH] [--prompt TEXT]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) model=${2:?missing model}; shift 2 ;;
    --resume) resume_id=${2:?missing session ID}; shift 2 ;;
    --list-models) list_only=1; shift ;;
    --base-url) base_url=${2:?missing base URL}; shift 2 ;;
    --state-root) state_root=${2:?missing state root}; shift 2 ;;
    --fixture-root) fixture_root=${2:?missing fixture root}; shift 2 ;;
    --prompt) prompt=${2:?missing prompt}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

if [ -z "${CODEX_PROVIDER_API_KEY:-}" ]; then
  printf '%s\n' 'Set CODEX_PROVIDER_API_KEY in this shell. It is injected only into the Codex child.' >&2
  exit 64
fi

model_args=()
if [ -n "$model" ]; then
  model_args=(--model "$model")
fi
python3 "$script_dir/list_models.py" --base-url "$base_url" "${model_args[@]}"
if [ "$list_only" -eq 1 ]; then
  exit 0
fi
if [ -z "$model" ]; then
  printf '%s\n' '--model is required after discovery; select an exact ID returned by GET /models.' >&2
  exit 64
fi
python3 "$script_dir/prepare_acceptance.py" --state-root "$state_root" --fixture-root "$fixture_root" --base-url "$base_url" --model "$model" >/dev/null

if [ -n "$resume_id" ]; then
  exec python3 "$script_dir/launch_codex.py" --state-root "$state_root" --fixture-root "$fixture_root" --model "$model" --resume "$resume_id" --prompt "$prompt"
fi
exec python3 "$script_dir/launch_codex.py" --state-root "$state_root" --fixture-root "$fixture_root" --model "$model" --prompt "$prompt"
