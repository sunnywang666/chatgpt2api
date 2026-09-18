# Isolated Codex CLI provider acceptance

This is an isolated Codex CLI acceptance package for the provider endpoint:

```text
https://app.hugsweetglobal.com/ai/codex/v1
```

Successful model discovery or deployment health does not prove that a real Responses turn can complete. Use the commands below for the approved account/API acceptance, and preserve the original session when the request outcome is unknown.

## 程序密钥允许端（第二阶段已上线）

工作台「ChatGPT / Codex → 密钥与 CLI → 创建程序密钥」勾选 **允许 Codex**，保存一次性显示的普通密钥，再按下方已有 CLI 步骤配置。也可以同时勾选「允许 Chat」。只允许 Chat 的密钥不能调用 Codex；具体编程、生图或其它该端已支持操作由应用请求决定，不再另设用途权限。模型发现成功不代表上游账号可执行。

「编辑允许端」保存后影响后续提交。并发修改返回409 `KEY_POLICY_REVISION_CONFLICT`，刷新重新核对，不自动覆盖。403 `KEY_ROUTE_DENIED` 表示密钥未允许实际请求的端；接口/工具未实现返回技术错误，如501 `SERVICE_OPERATION_UNAVAILABLE` 或 `NATIVE_TOOL_NOT_CLASSIFIED`，不是缺少生图或规划权限。撤销后返回401，原记录保留；未知请求不新建重试。

2026-09-17：端级政策已部署并获用户确认，两把保留旧key均允许Chat/Codex、三把旧key已撤销；新key按当前政策执行。模型查询成功不代表真实Responses授权已解决。新接入优先从工作台复制给AI配置，公开合同 /ai/integration.md 与复制文本同源，普通密钥另行私密提供。部署边界见 [部署指南](deployment.md#工作台程序密钥政策第二阶段)。

## What the launchers do

### Account login and authorization

Account JSON commits use a same-directory temporary file, file fsync, atomic replacement and directory fsync with 0600 permissions. A corrupt or unreadable pool fails closed instead of becoming an empty pool. If replacement may have happened, the original login reads back the durable account snapshot and exact credential digest; it does not repeat the token exchange. A confirmed non-write and an unresolved commit remain distinct results.

The account login/import flow is being added to the existing Workbench account
center. Its code, deployment and actual upstream acceptance are separate states;
do not treat this section as proof that the new UI has been deployed.

- **New account:** choose `接入账号` → `登录授权 Codex`. The service starts an
  official device-code login and the page shows a short-lived code and the official
  sign-in link. Complete sign-in at `https://auth.openai.com/codex/device` and enter
  that code. Workbench then reports the result without asking you to copy tokens.
- **Existing account:** choose `补充 Codex 授权` on the intended account. The service
  checks the returned upstream subject and workspace against that selected record.
  Another account is rejected; this never replaces the record's Chat credential,
  owner, ordinary keys, or original request/session bindings.
- **Already signed in locally:** choose `导入已有授权` and select the official
  `auth.json`. Only the allowlisted credential fields are submitted through the
  authenticated HTTPS management bridge. Do not paste the file into chat or logs.
  This fallback is also available when device login is unavailable for the account.
- **Chat authorization:** retain the existing controlled Chat token import path.
  Codex sign-in is not displayed as proof of Chat authorization or image capacity.

Device login is currently an upstream beta and may require enabling it in the
account's security settings or workspace permissions. The official sign-in page
is the only place to enter an OpenAI password or approve OpenAI authorization.
[Official authentication documentation](https://learn.chatgpt.com/docs/auth#login-on-headless-devices).

Closing the dialog hides it; cancelling explicitly stops the local login operation.
Browser refresh resumes the saved opaque operation identifier, not a second login.
Expired or interrupted operations are shown explicitly. A failed or uncertain
exchange is not automatically repeated. `授权已保存，实际调用待验证` means only that
credentials were saved; real Codex tool execution still requires the isolated
client acceptance below.

The production management URL prefix is `/api/v1/content/ai-service` (the Content
service's configured API prefix), not the public `/ai/codex/v1` program endpoint.
The existing boss/administrator same-origin attachment route is
`POST /api/v1/content/ai-service/pool/codex-authorization`. Ordinary program keys
cannot manage accounts. The UI supplies a stable opaque `account_ref` when
attaching to a selected service-pool record; `pool-row-N` is display-only.
The legacy no-reference attachment remains compatible with its strict unique
subject/workspace match. Login/import requests keep the same role and ownership
boundaries and never return upstream access, refresh or ID tokens to the browser.

`examples/codex_client/run-codex.sh` and `examples/codex_client/run-codex.ps1` first request `GET /models`. Discovery accepts native Codex `models[].slug` and OpenAI-compatible `data[].id` responses without changing the provider protocol. Run discovery without a model, then select one exact returned ID before they create an isolated state root and launch Codex.

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

A Codex upstream 401 keeps that credential in `auth_required`, including after a service restart or a successful model/usage refresh. Those read endpoints do not prove that Responses execution is authorized. A changed access token or ChatGPT account identifier can be checked again; the service does not convert a Platform login into Codex authorization or replay the rejected request. Existing Content account credentials and task bindings are retained. Requests capture their authorization before dispatch, and failure updates compare it under the account rotation lock so an old response cannot reject replacement credentials. Legacy 401 records without a credential fingerprint remain unbound; a witnessed rotation or a token issued after the recorded failure permits fresh observation, while ambiguous history remains authorization-required without inventing a fingerprint.

The candidate serves HTTP Responses and SSE only; WebSocket transport and Codex Desktop are not yet accepted. Account observations expire after five minutes and are refreshed with bounded read-only probes. A missing/failed observation is not zero remaining quota. Upstream use remains subject to the account's real limits and availability; this service adds no budget, top-up, or internal token allocation.

A pending or unknown session is retained and rejected with `codex_session_outcome_unknown` on continuation. A POST HTTP 408, like a transport timeout or 5xx, has an unknown outcome and is never automatically replayed. A terminal SSE response already parsed before client disconnect remains terminal. Do not create another session to replay the uncertain request. There is not yet an operator UI for resolving that receipt. This candidate keeps up to 256 session bindings and 256 response-owner references per account in the existing account record; it never evicts a prior owner to accept new work. `codex_binding_capacity` requires operator handling and is a storage protection, not an allowance of requests or tokens. Long-term retention/archival handling remains a release consideration.

`codex_limited` means the selected upstream reported a limit. Existing sessions stay on their original account; a new independent session may use another eligible account. `codex_auth_required` means the upstream rejected the account credential with HTTP 401; inspect its authorization and refresh state before an authorized refresh or re-import. It does not prove that the credential merely expired. `codex_access_denied` means upstream HTTP 403: inspect account access and the request conditions instead of repeatedly refreshing the token. These two errors are returned as HTTP 503 by the service because its ordinary caller key has already authenticated; they do not mean that the caller key needs replacing. Neither condition permits replaying an uncertain request. These operational limits must be considered alongside the pending real-upstream and Windows checks before calling the service generally available.

An account already stored in the pool must not be added again to investigate a Codex rejection. Check the original account's saved authorization, forwarded account identity, credential refresh time, rejection, and session binding first. The existing password-login path obtains Platform OAuth authorization; its successful login, model discovery, or usage read does not establish Codex execution authorization. A fresh official Codex login is a fallback only when the existing authorization cannot be used. Preserve the original account and session when applying an explicitly authorized replacement; importing a different access token as a new account does not transfer that binding.

For existing Web/Platform accounts, the normal account refresh now retains a valid UUID returned by the authenticated default-account endpoint when the stored account ID is empty. It does not overwrite a different stored workspace or clear an ID on a missing/invalid response. The write is checked against the credential actually sent, so an old response cannot update rotated credentials. Existing rejection records and session bindings remain intact. After an account-identity header really changes, refresh the Codex observation through the existing route and validate one authorized continuation of the original session; a successful metadata read alone does not prove that continuation will work.

The native model catalog updates the same account's safe model observation without clearing a Responses rejection or refreshing the quota timestamp. Selection retains the existing freshness window and bounded account probes; an arbitrary missing model must not force repeated refreshes of fresh observations. `available_accounts` remains a count of accounts with fresh eligible observations, not a count of proven successful Responses executions.

Image quota is a separate `limits_progress.image_gen.remaining` observation. Missing or invalid values remain unknown (`null`), while an observed `0` means zero remaining. Unknown quota does not admit image work. Codex usage windows retain upstream limit IDs, percentage used, window seconds, reset timestamps, and observation status; they are not image counts or per-request charges.

These are API data semantics. The existing Workbench capacity projection distinguishes unknown from zero. The Provider's older embedded administrator pages still coerce unknown quota to zero in some cells and totals; their display correction is outside the current phase's UI freeze. Do not use those legacy totals as proof of complete available capacity.
