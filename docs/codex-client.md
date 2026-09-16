# Isolated Codex CLI provider acceptance

This is an engineering acceptance package for the planned provider endpoint:

```text
https://app.hugsweetglobal.com/ai/codex/v1
```

The endpoint is not deployed at the time this package was written. The commands below are for the approved future account/API acceptance; they do not establish that the service is live today.

## What the launchers do

`examples/codex_client/run-codex.sh` and `examples/codex_client/run-codex.ps1` first request `GET /models`. Run discovery without a model, then select one exact returned ID before they create an isolated state root and launch Codex.

The state root contains a generated `config.toml` with this custom Responses provider shape:

```toml
model_provider = "hugsweet-codex"

[model_providers.hugsweet-codex]
base_url = "https://app.hugsweetglobal.com/ai/codex/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
```

The bearer key is absent from that file. `launch_codex.py` reads `CODEX_PROVIDER_API_KEY` from its own environment, removes it, and calls `execvpe` with a fresh Codex environment containing only `OPENAI_API_KEY`; the key is never placed in Codex argv. The scripts do not copy a normal `CODEX_HOME`, login, or authentication file. They do not put a key in a command argument, shell history, output file, or debug log. Avoid `--debug` during this flow because debug logging can contain HTTP details and bodies. Codex documents custom `model_providers`, `env_key`, `wire_api`, and per-provider retry controls in its advanced configuration reference. [Codex advanced configuration](https://developers.openai.com/codex/config-advanced/)

The custom provider uses `request_max_retries = 0` and `stream_max_retries = 0` so an acceptance failure is reported to the operator. The launcher does not implement a high-level retry. It does not request or use any admin capability.

Model discovery rejects redirects instead of forwarding the bearer key to a redirect destination. Use the final HTTPS provider Base URL directly. Plain HTTP is accepted only for the local mock hosts `localhost`, `127.0.0.1`, and `::1`; other hosts, embedded credentials, queries and fragments are rejected before discovery or configuration creation.

## macOS acceptance

Obtain the ordinary provider key from the approved local secret source without echoing it. The following prompt keeps the value out of shell history and terminal output. Select the model after the list preflight:

```bash
export CODEX_PROVIDER_API_KEY="$(python3 -c 'import getpass; print(getpass.getpass("Provider key: "))')"
bash examples/codex_client/run-codex.sh --list-models
bash examples/codex_client/run-codex.sh --model <exact-id>
```

The second command creates `examples/codex_client/.codex-client-state` and the disposable `acceptance-fixture`. It asks the actual Codex CLI to repair the intentional `multiply()` defect in `calculator.py` and run `python -m unittest -v`. Its JSONL output starts with a `thread.started` event containing a `thread_id`. Preserve that ID for the second-turn check:

```bash
bash examples/codex_client/run-codex.sh --model <exact-id> --resume <thread_id> \
  --prompt 'Run python -m unittest -v again. State the exact changed file and the test result.'
unset CODEX_PROVIDER_API_KEY
```

Do not pass `--resume --last`: the explicit ID prevents a normal personal session from being selected. Use a new `--state-root` for a separate acceptance. The launcher refuses to replace a non-generated `config.toml` in that root.

## Windows external-machine acceptance

On the Windows machine, read the key without echoing it or placing it in the command line:

```powershell
$secureKey = Read-Host "Provider key" -AsSecureString
$env:CODEX_PROVIDER_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
.\examples\codex_client\run-codex.ps1 -ListModels
.\examples\codex_client\run-codex.ps1 -Model <exact-id>
```

For the second turn, use the `thread_id` from the first JSONL output:

```powershell
.\examples\codex_client\run-codex.ps1 -Model <exact-id> -Resume <thread_id> `
  -Prompt "Run python -m unittest -v again. State the exact changed file and the test result."
Remove-Item Env:CODEX_PROVIDER_API_KEY
```

The PowerShell launcher invokes the same Python `execvpe` bridge. The bridge removes inherited `CODEX_HOME`, `OPENAI_API_KEY`, and `CODEX_PROVIDER_API_KEY`, then gives only `OPENAI_API_KEY` to Codex. It requires PowerShell 7, Python 3, and Codex CLI on `PATH`.

## Local deterministic tool-cycle mock

No real provider request is needed to check client routing. Start the included mock in one terminal:

```bash
python3 examples/codex_client/mock_provider.py
```

In another terminal, use a fake key and its loopback endpoint:

```bash
CODEX_PROVIDER_API_KEY=fake-key-not-a-secret \
  bash examples/codex_client/run-codex.sh --base-url http://127.0.0.1:18787/v1 \
  --model gpt-5.1-codex-mini
```

The mock prints a redacted record. With Codex CLI 0.149.1 on macOS, the observed first agent request was `POST /v1/responses` with `Accept: text/event-stream`, `Content-Type: application/json`, and `Authorization: Bearer <redacted>`. Its JSON fields were `client_metadata`, `include`, `input`, `instructions`, `model`, `parallel_tool_calls`, `prompt_cache_key`, `reasoning`, `store`, `stream`, `tool_choice`, and `tools`; the request has `stream: true`. The client also supplied Codex correlation metadata headers (`x-codex-window-id`, `x-codex-turn-metadata`, `x-client-request-id`, `session-id`, and `thread-id`).

The mock emits `response.created`, `response.in_progress`, `response.output_item.added`, `response.function_call_arguments.done`, `response.output_item.done`, and `response.completed` as Server-Sent Events with `content-type: text/event-stream`. It calls the CLI's advertised `exec_command` function three times: read the fixture, change `return left + right` to `return left * right`, then run `python -m unittest -v`. Each successor request must return the matching `function_call_output` call ID before the mock emits the next call. Run the explicit `--resume <thread_id>` command while that mock is still running to prove the isolated stored session can accept a second turn.

This proves the installed CLI's local function-call transport and executor cycle, including a file edit and test. The calls are deterministic mock output, not provider-model behavior. A deployed-provider run is still required to prove that the provider faithfully streams a real model's tool calls and produces the same changed fixture plus test result.

## Candidate router/service engineering QA

`candidate_chain_qa.py` is a local Mac-only engineering harness for the candidate implementation. It starts the included mock upstream, injects a session factory that maps only the candidate service's fixed native upstream URLs to that loopback server, then starts the real `api.codex` router and `CodexService` on another loopback port. Production code receives no URL override.

Run it with a Python environment containing the project's FastAPI, Uvicorn, and curl-cffi dependencies:

```bash
python3 examples/codex_client/candidate_chain_qa.py
```

The harness uses a temporary `JSONStorageBackend`, one fake Codex account, and fresh mock observation data. It creates a disposable ordinary `AuthService` key, runs the real local Codex CLI through model discovery, relays the mock read/edit/test function cycle, and resumes the exact session. It then revokes that ordinary key and verifies `GET /models` and `POST /responses` return 401; a separate local admin key receives 403. Reports contain IDs and status codes only and are written to the ignored `data/codex-client-qa/` directory.

This is engineering mock evidence only. It does not contact a real provider, use an account credential, establish deployment state, or prove production-provider acceptance.

## Evidence boundaries

The local mock is engineering evidence. A successful real CLI fixture run proves the selected provider account can serve that command and Codex can edit the disposable fixture and run its test. A Windows run is independent external-machine evidence. Neither result proves Codex Desktop behavior, cloud-task features, administrative APIs, or any feature outside the CLI/Responses path.


## Current operational limits

The candidate serves HTTP Responses and SSE only; WebSocket transport and Codex Desktop are not yet accepted. Account observations expire after five minutes and are refreshed with bounded read-only probes. A missing/failed observation is not zero remaining quota. Upstream use remains subject to the account's real limits and availability; this service adds no budget, top-up, or internal token allocation.

A pending or unknown session is retained and rejected with `codex_session_outcome_unknown` on continuation. A POST HTTP 408, like a transport timeout or 5xx, has an unknown outcome and is never automatically replayed. A terminal SSE response already parsed before client disconnect remains terminal. Do not create another session to replay the uncertain request. There is not yet an operator UI for resolving that receipt. This candidate keeps up to 256 session bindings and 256 response-owner references per account in the existing account record; it never evicts a prior owner to accept new work. `codex_binding_capacity` requires operator handling and is a storage protection, not an allowance of requests or tokens. Long-term retention/archival handling remains a release consideration.

`codex_limited` means the selected upstream reported a limit. Existing sessions stay on their original account; a new independent session may use another eligible account. `codex_auth_required` requires an authorized account refresh or re-import. Neither condition permits replaying an uncertain request. These operational limits must be considered alongside the pending real-upstream and Windows checks before calling the service generally available.
