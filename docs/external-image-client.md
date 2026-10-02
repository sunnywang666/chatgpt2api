# Persistent image-task client

## 中文接入要点

第二阶段双端权限已于2026-09-17部署并由用户确认。图片/文字程序勾选「允许 Chat」，需要 Codex 时可另勾选「允许 Codex」；同端操作没有用途子权限。修改允许端不删除原回执；撤销后认证失效。发布时两把保留旧key均双端，三把旧key保持撤销，新创建密钥以其当前政策为准。

首选入口是工作台「ChatGPT / Codex → 密钥与 CLI → 复制给AI配置」。复制说明不包含真实密钥，交给负责目标应用的AI；密钥通过本地私密环境配置另行提供。公开同源合同位于 https://app.hugsweetglobal.com/ai/integration.md ，不需要网站后台登录。下文为已有Python客户端的备用操作说明。第三阶段公共Chat增量的真实部署/接入证据单列，不以第二阶段发布推断第三阶段已经验收。

- `SERVER_ROOT` 是服务根（`https://app.hugsweetglobal.com/ai`）；兼容 API Base URL 另加 `/v1`。持久任务不能拼成 `/v1/api/...`。
- 在工作台“AI 服务连接”中创建普通调用密钥，仅通过私有渠道交付，填入本地权限为 `600` 的环境文件。它没有账号池管理员能力；撤销后原密钥立即失效。
- 用下面的 `submit` 保存原任务号，再用 `status` 查询、`download` 保存图片。编辑时加 `--image 本地图片`，可传多张参考图。
- 进程退出、网络超时或结果未知后，继续使用原状态文件和任务号。客户端不会自动换号或重新生成。
- 可靠共用候选完整受理后先持久保存实际输入；池满返回原 `queued` 回执，新增/恢复账号后服务自动派发，服务重启也不要求重提。`waiting` 描述当前原因。部署前后的行为以实际版本为准，见[部署说明](deployment.md#可靠共用与动态容量候选2026-09-20)。
- 公司账号界面由有权限的成员管理公司统一账号，普通调用者通过自己的程序 Key 使用，显示上游观测值与更新时间；未知、真实零、读取失败和调用后待刷新分开。没有内部预算或张数分配。
- 同源工作台界面无需 CORS。跨域浏览器调用默认关闭；管理员需将明确来源写入 `CHATGPT2API_CORS_ORIGINS`（逗号分隔）或配置 `cors_origins`，不开放凭据通配来源。普通后台 HTTP 程序不受 CORS 影响。
- 本接入包不等于小乐的程序已接通。她的调用位置确定后，才能核对 SDK、DSH 或其它调用方式的最小适配。

`examples/image_client.py` is a dependency-free Python client for an external caller that needs a durable image-task receipt. It uses only the Python standard library. Its local state contains no bearer token, prompt text, or image bytes.

This is an engineering client for the persistent task API. The existing public service root is `https://app.hugsweetglobal.com/ai`. Local fixture tests do not establish a reachable endpoint, a usable provider account, or a successful external image generation.

## 暂停某个原请求的结果查询（候选能力）

对文字调用 `POST /api/chat-requests/{request_id}/recovery-control`，对图片调用
`POST /api/image-tasks/{client_task_id}/recovery-control`，请求体是
`{"state":"paused"}` 或 `{"state":"active"}`。既有内部文字调用者使用
`/api/conversation-bindings/text-requests/{request_id}/recovery-control`，仍按原认证身份归属。
公司入口沿现有公司会话转发同一操作，不需要员工提供 Key。

独立客户端复用原状态文件：

```sh
python3 examples/image_client.py --env-file .image-client.env chat-recovery-pause --state ./my-chat-request.json
python3 examples/image_client.py --env-file .image-client.env chat-recovery-resume --state ./my-chat-request.json
python3 examples/image_client.py --env-file .image-client.env recovery-pause --state ./my-image-task.json
python3 examples/image_client.py --env-file .image-client.env recovery-resume --state ./my-image-task.json
```

- `recovery_control.state=paused` 停止后续自动读取；普通查询和原结果 recover 也不会暗中绕过。
  文字公开回执位于 `recovery.control`，图片及内部文字回执位于 `recovery_control`。
- `pausing` / `in_flight=true` 表示已有本地操作仍在途，不能宣称它已经取消。它可结束并保存原结果，
  后续读取不会再启动。当前已开始的有界读取可能包含多次上游 HTTP；暂停不是中途断网。
- 状态持久化、重复操作幂等。恢复不发送模型请求、不清空原输入/结果/错误/UNKNOWN、不绕过
  `Retry-After` 和已有退避，也不解除工作流暂停或其他停止标记。
  请求恢复到 active 时，若仍有内部停止标记，返回 `state=stopped, operator_stopped=true`，客户端明确报未恢复。
  请求暂停时，`paused/pausing` 表示本次暂停意图，`operator_stopped=true` 同时说明还存在另一个停止范围。
- UNKNOWN 的账号占用和同会话顺序保护保留。关联自动纠正的未发送子请求也不能在暂停后越过发送边界。
  该控制不是取消一般排队生成的接口，不会把未知结果改写成失败或成功。
- 不提供“清空这些 UNKNOWN”的删除接口。删除本地记录不能终止上游，也会失去原结果读回、
  账号会话归属和顺序证据；普通历史保留策略不得删掉暂停中的请求。这里只管理结果查询，
  不自动隐藏待处理项、不宣称上游已停止。

此能力须在匹配候选部署后才线上生效；本地与受控 HTTP 验证不是生产已应用。

## Server requirements

Set `SERVER_ROOT` to the service root used by the external caller. The root may include an ingress prefix, but it must not end in `/v1`. The example environment contains the existing `/ai` service root and no credential. Do not reuse an OpenAI SDK `OPENAI_BASE_URL` value ending in `/v1`: doing so would incorrectly produce `/v1/api/...` paths.

The example environment file contains no credential:

```sh
cp examples/image-client.env.example .image-client.env
chmod 600 .image-client.env
# Edit .image-client.env locally.
```

The server must expose these authenticated routes under `SERVER_ROOT`:

| Purpose | Method and path |
| --- | --- |
| Discover the current model catalog | `GET /v1/models` |
| Submit one persistent generation | `POST /api/image-tasks/generations` |
| Submit one persistent edit | `POST /api/image-tasks/edits` |
| Read original receipts by caller ID | `GET /api/image-tasks?ids={client_task_id}` |
| Continue polling an eligible original receipt | `POST /api/image-tasks/{client_task_id}/resume-poll` |
| Download one receipt-owned result | `GET /api/image-tasks/{client_task_id}/images/{index}` |
| Synchronous-compatible generation | `POST /v1/images/generations` |
| Synchronous-compatible edit | `POST /v1/images/edits` |

Every request uses `Authorization: Bearer $CHATGPT2API_BEARER_TOKEN` and the fixed public-service marker `X-Workbench-Image-Client: 1`. The list endpoint must return only receipts owned by that ordinary bearer identity and must account for each requested ID in either `items` or `missing_ids`. The download endpoint must enforce the same ownership from the original receipt and return image bytes. The client constructs this authenticated receipt route itself; it does not follow a raw `/images/...` URL or a storage/container URL from task data.

The client depends on two details that older server baselines may not provide:

- immutable input drift for an existing `client_task_id` must surface as HTTP 409;
- the authenticated receipt-owned `GET /api/image-tasks/{client_task_id}/images/{index}` route must exist.

The public external image service accepts only `gpt-image-2`, one output per task (`n=1`), and `b64_json` for synchronous-compatible calls. Codex image routing and Codex recovery are internal capabilities and are not exposed here. The external interface does not accept caller-supplied provider bindings, account identities, or conversation IDs. Provider accounts, raw image storage, files, admin routes, and account/key management remain outside the public `/ai` surface.

For this Chat image route, `size` and `quality` are instructions added to the upstream prompt, not structured controls that guarantee exact output pixels or a quality tier. The caller must inspect the saved image's actual dimensions; a `1024x1024` request can return a different square size. Leaving them unset / `auto` uses the existing upstream defaults. Exact pixel sizing is not a supported capability of this route.

Execution currently reuses the service's existing paid-account conversation-binding route. Free accounts are not eligible for this route. A response saying that no eligible image resource was admitted describes the route at that moment; it is not evidence that every account in the private pool has exhausted an upstream quota.

The synchronous-compatible routes use the same persistent receipt machinery as `/api/image-tasks`. Supply a stable `client_task_id` on every synchronous request. A completed request returns HTTP 200 with `b64_json` data and the original task ID. Queued/running requests or requests awaiting authoritative recovery return HTTP 202 with that same ID. With durable admission enabled, accepted requests wait durably for account capacity; an older retained resource-admission failure may still return HTTP 429, and other known terminal failures return HTTP 502. Transport/body-reader 429 before acceptance is distinct from upstream 429: inspect `rate_limit.layer`, phase, request ID and Retry-After. Never create a new ID automatically after timeout or non-200. Poll original receipts; generated-but-undownloaded results resume downloads only.

## Commands

The client reads the env file only when `--env-file` is supplied. Global options precede the command.

Read the service model catalog before submitting. For image output select `gpt-image-2`; text-capable entries have `capabilities` including `text` and `image_input`. A row may also include `accounts` capability observations with opaque `account_ref` values plus state, reason, and capabilities. Those observations do not promise an immediately available slot or upstream quota; admission is decided when a request is submitted:

A successful account observation remains capability evidence if a later refresh fails or the account becomes limited. Its account row then reports `observed_at` and `observation_state` (`observed`, `read_failed`, `stale`, or `unknown`) with the existing reason such as `read_failed`, `stale`, or `limited`; it is not selected for a new send until refreshed successfully. When other known text models remain, the directory returns them with `model_catalog: {"state":"partial"}`. A request for a model absent from an incomplete paid catalog returns retryable `503 MODEL_DISCOVERY_UNAVAILABLE`, rather than claiming permanent unsupported. Catalog reads use a bounded shared worker set and timeout budget, so a slow account is retained as non-routable evidence and retried without serially blocking the directory. Successful accounts retain their own 300-second cache while failed accounts and anonymous discovery retry independently; credential or account-state changes invalidate executable reuse. Anonymous model discovery preserves numeric and HTTP-date Retry-After on 429, and authenticated reads retain the existing account pacing cooldown.

`gpt-image-2` is the Provider image-generation alias, not proof that upstream `/models` returned that exact name. Its account rows come from saved `limits_progress.image_gen` observations. Zero remaining quota, a later read failure, or staleness retains capability history but does not claim current dispatch capacity; absent or invalid observations are not inferred from subscription type or account presence. Image generation admission and send-time quota checks remain separate from this directory projection.

```sh
python3 examples/image_client.py --env-file .image-client.env models
```

Submit a generation. `--model gpt-image-2` is required. The durable client generates and saves a caller-owned task ID before the first POST when `--client-task-id` is omitted:

```sh
python3 examples/image_client.py --env-file .image-client.env submit \
  --state ./my-image-task.json \
  --prompt 'A concise description of the requested image' \
  --model gpt-image-2
```

Add one or more local `--image` arguments to submit an edit as multipart form data. The server accepts at most 16 input images, at most 50 MiB per image, and at most 100 MiB combined:

```sh
python3 examples/image_client.py --env-file .image-client.env submit \
  --state ./my-edit-task.json \
  --prompt 'Apply the requested edit' \
  --model gpt-image-2 \
  --image ./input-front.png \
  --image ./input-detail.jpg
```

Before either submit request is sent, the client atomically stores the `client_task_id`, a canonical input fingerprint, content hashes and sizes for local images, and lifecycle phase. It does not store the prompt itself. Re-running `submit` with the same state and identical inputs performs a status query; it does not POST again. Reusing that state or task ID with different input fails locally. To intentionally create another image, use a different state file and task ID.

A caller-supplied task ID must contain 1 to 200 ASCII characters chosen from letters, digits, `_`, `.`, `:`, and `-`. The client validates the same contract before writing state or making a request.

Query the original receipt without changing upstream polling:

```sh
python3 examples/image_client.py --env-file .image-client.env status \
  --state ./my-image-task.json
```

An explicit original ID can be used when no state file is present:

```sh
python3 examples/image_client.py --env-file .image-client.env status \
  --state ./absent-state.json \
  --task-id ORIGINAL_CLIENT_TASK_ID
```

`status` is a read-only receipt lookup. It does not ask the provider to keep polling. `resume` is a POST that tells the server to continue polling the already submitted receipt. It does not submit a new generation and is valid only when the server has retained an eligible original unknown-outcome conversation boundary:

```sh
python3 examples/image_client.py --env-file .image-client.env resume \
  --state ./my-image-task.json \
  --extra-timeout-secs 30
```

Download result index zero after the receipt reaches `success`:

```sh
python3 examples/image_client.py --env-file .image-client.env download \
  --state ./my-image-task.json \
  --index 0 \
  --output ./result.png
```

The client refuses to overwrite an existing output, rejects missing or non-file image inputs, caps each local input image at 50 MiB, caps a downloaded result at 100 MiB, and refuses redirects that change origin so that the bearer credential cannot be forwarded to another host. The server additionally enforces the 16-image and 100 MiB combined edit limits described above.

Download completed images promptly. The existing server image-retention policy still applies to stored files; the external task receipt remains to prevent the same ID from generating again after file expiry. An expired result may therefore return 404 without creating a replacement image.

## Failures and recovery

The client prints JSON to stdout on success and a JSON error to stderr on failure. Common server responses are:

- `400`: malformed input or a resume request that is not valid for the receipt;
- `401`: missing, invalid, or expired bearer credential;
- `403`: the ordinary identity is authenticated but not permitted to perform the operation;
- `409`: the caller reused a durable task ID with different immutable input;
- `429`: classify the source before acting: company transport protection (`company_transport`), Provider technical protection (`provider_capacity`), or real upstream ChatGPT/Codex rate limits (`upstream_chatgpt` / `upstream_codex`). Retain the original request ID, `rate_limit` origin/phase/request IDs and Retry-After/cooldown evidence. A busy company entrance must not cool down a ChatGPT account; an upstream limit does not authorize resubmitting an UNKNOWN request on another account;
- `5xx`: the server or its upstream path failed.

An HTTP response is recorded in the local state. A timeout, disconnect, invalid response, or interruption after the state was prepared has an unknown submission outcome. The client never converts that uncertainty into a new ID and never silently retries the submit POST. Restart with the same state and run `status`, or run the identical `submit` command, which performs the same status lookup. If the server reports that the original ID is missing, the client stops; deciding to create a different task requires an explicit new state file and ID.

Do not put the bearer token in command arguments, task state, logs, screenshots, or source control. Keep the filled env file private.

For the authorized joint observation window, the existing `scripts/audit_pacing_log.py` reads a bounded local log export from stdin. Current clock events retain an opaque original-request reference, model, operation, persisted input size, send sequence, layer/phase and rate-limit evidence in the report; old missing fields stay unknown. Upstream header IDs are hashed, and `coverage.omitted_samples` discloses any sample truncation. This script makes no network calls. A message-start event occurs before final send guards and is not proof of a completed upstream send. Correlate it with the original receipt/result and an observed pool snapshot: current clock logs alone do not record concurrent occupancy or trusted source, and cannot establish a safe concurrency or interval.


Codex coding is a separate route (`/ai/codex/v1`) documented in [the Codex CLI package](codex-client.md). It does not replace these image-task endpoints or change their supported image model. Its deployment and real CLI acceptance are tracked separately.

## 公共Chat文字与图片理解（第三阶段增量）

同一个现有客户端增加 `chat-submit`、`chat-status`、`chat-recover`，不需要另装SDK。模型从 `models` 的实时目录选择具有text能力的ID；不能用图片模型或未经公布的auto别名。请求与错误完整合同以页面复制文本和公开 `/ai/integration.md` 为准。

```sh
python3 examples/image_client.py --env-file .image-client.env chat-submit \
  --state ./my-chat-request.json --model DISCOVERED_TEXT_MODEL \
  --prompt '仅描述这张商品图能确认的事实，看不清的留空。' --image ./input.png
python3 examples/image_client.py --env-file .image-client.env chat-status --state ./my-chat-request.json
# 只向上游读取原请求结果，不重发生成：
python3 examples/image_client.py --env-file .image-client.env chat-recover --state ./my-chat-request.json
```

纯文字省略 `--image`。本地图片转为data URL字节，不把路径传给服务器。提交前原子保存request_id及输入指纹，状态文件不存密钥、提示词或图片字节；相同输入再次执行只查询，漂移在本地拒绝。收到202、断线或超时后退出再运行status，绝不新开ID自动重试。404表示未在当前身份找到原回执，需要调查，不证明可以重发。

完整受理后的原请求及实际输入由服务器持久保存。满载时返回202并以原ID等待；新增符合条件的账号、账号恢复或原占用释放后自动派发，不要求客户端换Key、重启或重提。服务重启后尚未发送的请求沿持久输入继续，已发送/UNKNOWN仍绑定原账号查结果。等待不再受旧32个内存任务上限拒绝。请求体读取仍有2个并行读取及执行输入256MiB等技术保护，和账号额度、会话数量分开；429 `CHAT_BODY_READER_CAPACITY_EXCEEDED` 属于受理前读取繁忙。旧版 `failed` / `TEXT_TASK_CAPACITY_EXCEEDED` / `recovery.upstream_outcome=not_sent` 回执才允许退避后原ID原输入再次POST。示例客户端保持查询优先，不自动执行这个重试，也不能删除状态文件换ID绕过保护。`waiting.next_check_at`、`recovery.next_at`是退避证据，不是客户端重新提交指令。

原生持久 Chat 文字请求的本地执行等待有界：受理至少15分钟、此后至少3次符合条件的原结果查询仍无法取得结果，且没有活跃执行或恢复领取时，可返回 `failed` / `RESULT_UNRECOVERABLE`，释放给其他独立会话的执行位置。`recovery.upstream_outcome=unknown` 仍表示上游结果尚未确认，不能按“未发送”处理；`recovery.retryable` 仅用于原ID结果查询，不授权再提交模型。保留原请求编号与输入，迟到的准确结果可更新为成功；同会话下一轮仍需原结果确认。网络超时、授权失败、429或查询次数不足不会单独触发此处置。此规则不代表图片/兼容转发/Codex具有相同超时策略。

Chat成功状态是 `succeeded` 且有content；图片任务成功状态仍为 `success`，不要混用。只有实际取得的上游usage才可报告，当前未返回usage时应显示未知。现有公共Chat不提供工具执行或结构化输出保证；应用自行校验模型文本，不把JSON解析失败包装为成功。


### 连续工作会话（sequential-v1 增量）

原生 Chat 文字的每个公共回执带脱敏 `execution`。`send_state=not_sent` 需要明确未发送证据，`attempted` 只证明进入发送边界，`response_received` 只证明收到 HTTP 响应头；均不能单独证明生成成功。`sent_at`、`response_received_at`、`local_finished_at` 分别记录发送、响应、本地执行结束。`first_stream_event_at` 是首个流事件，不是思考或首个文字 token；旧回执缺发送证据时返回 `unknown`。不要将 `running` 一律展示为“正在生成”。 `stream_end`区分done（收到SSE结束标记）、eof、hard_timeout、transport_error与consumer_closed；`stream_ended_at`、`sse_data_count`、`sse_parse_errors`及`sse_error_event`只保存结构诊断，不包含原始正文或错误内容。done和HTTP 200都不证明生成完成：兼容流也要读回原分支的最终文本。缺尾文本仅从核验的原结果补回，空结果保持UNKNOWN；已发文本与最终结果冲突时保留旧流，原ID恢复读取生成新的完整结果文件，不再发送模型请求。历史回执缺诊断字段不补猜。 原分支仍未确认、但同一明确当前链已存在后续用户轮次及完成回答时，recovery.reason=REQUEST_CONVERSATION_ADVANCED：显示“原请求无最终结果，会话已被外部继续”，结束原请求前台等待。后续回答不归原ID，不覆盖原输入/游标，不证明原请求已取消，不回退到旧位置；若满足下文的空回复自动重试条件，沿最新已完成位置追加原输入，保留人工后继及原UNKNOWN证据。

`observation_started_at` 是首次有效原结果观察，`last_checked_at` 是最近有效观察，`last_progress_at` 仅在同一原请求的有效消息结构、状态或文本长度发生变化时出现。原分支的已知 `code`、`execution_output`、`thoughts`、`reasoning_recap` 封装也可观测，只保留类型、长度和条目计数，不持久化或公开正文；未知／媒体封装不计为有效空结果。上游 `update_time` 单独变化、无关分支变化、网络／429／授权失败不算进展。对持续 `REQUEST_RESULT_INCOMPLETE`，至少 15 分钟无可验证变化且至少 3 次有效无变化读后，持久化 `phase=stalled / wait_state=ended / wait_ended_at`：本次结果等待异常结束，原 `status=unknown` 和上游结果未知仍保留。此标记本身**不释放**账号 turn，也不放行任意同会话后续；已确认空回复的唯一关联重试按下文转交原占用，它与已有 `RESULT_UNRECOVERABLE` 合格无结果释放规则分开。迟到结果仍沿原 ID 接回，重启不重置观察或结束记录。

客户端读到 `wait_state=ended` 应停止本次前台等待，保存原 ID、输入和最后回执，显示未确认的原因、`recovery.attempt/reason/next_at`，旧策略以后仍沿原 ID 恢复；下文新请求的有界处置结束旧尝试后停止自动查询，保留证据并显示接续结果或确切阻塞。`resources.local_worker/account_turn/conversation` 分别解释本地执行、账号执行位置、会话保护；工作槽位和归档以 `/work` 回执为准。示例显示：“请求已发送并收到响应；本地执行已结束；原结果连续无可验证进展，本次等待结束；上游结果未知，账号执行位置仍占用，同会话后续受保护。尚未确认空回复时禁止自动重发；确认后显示 completion 的关联重试和结果。” 这是工程状态合同，不是对生产或真实上游已恢复的声明。

持续推进的应用可在现有 `/api/chat-requests` 请求中提供 `client_conversation_id`；这是调用者的工作会话引用，不是上游 ChatGPT conversation ID。每轮使用不同的 `client_request_id`，除首轮外传 `previous_request_id` 指向已受理的同会话原请求。每次只提供新增输入/工具结果与必要附件，不把全部旧历史再次追加。服务在原 owner 范围和原 SQLite 插入事务中核对前序、唯一后继，并沿前轮实际账号、上游会话和已证明最终回答承接；不允许客户端传管理绑定或原始上游游标。

回执增加 `conversation={protocol:"sequential-v1",client_conversation_id,previous_request_id}`，不暴露池账号/上游ID。前序为queued/running时可以预受理后轮，后轮持久排队，直到前序succeeded并核验原承接位置才实际发送；前序unknown/failed返回409 `CHAT_PREVIOUS_REQUEST_PENDING`。漏前序、跨会话或已有后继返回明确409；这些拒绝不代表新请求已受理。依赖前轮结果才能构造输入的步骤须先等结果，不能用占位输入提前提交。重复同ID同输入仍返回原回执，即使会话已推进到更后面；旧不带此字段的请求保持原哈希及单次调用方式。

唯一例外是原请求已发送且原账号读回确认最终轮次 `end_turn=true`、文字为空：原回执继续为 `unknown`，并显示 `terminal_empty={verified:true,original_request_id,observed_at,same_conversation_continuation:true}`。应用若确需纠正协议，可用**新的** `client_request_id`、相同 `client_conversation_id`、指向该原请求的 `previous_request_id`、新的纠正说明作为末尾 user 消息，并显式传 `continue_after_terminal_empty:true`。服务原子限制该前序只有一个后继，绑定原账号、原上游会话与空回答的最终节点，发送前重新读取原结果；证据不符、读取失败或会话位置已变时不发送纠正轮次。此字段不表示原业务请求失败或可重发，也不允许绕过其他 `unknown`。普通 CLI 接受succeeded/queued/running前序，不会自动把UNKNOWN变成纠正轮次。

若这条**纠正请求本身**已受理，却在发送前以 `failed`／`CHAT_TERMINAL_EMPTY_UNVERIFIED`／`recovery.upstream_outcome=not_sent` 结束，升级或普通状态读取不会自动重发。核实原 ID、未发送时间线和持久输入后，可显式 `POST /api/chat-requests/{纠正请求ID}/recover`，请求体为 `{"resume_unsent_correction":true}`。服务仅在同一调用身份、原输入哈希、原账号/会话/消息位置和新鲜空终态证明都通过时，将**同一请求 ID** 原子放回原等待队列；派发前仍再次读取原结果。证据变化、输入缺失或曾开始发送均拒绝，不创建新 ID，也不修改原 `unknown`。默认 `{}` 的 recover 仅查询原结果，不重新发送模型请求；客户端不得自动给所有 failed/UNKNOWN 请求加此标志。

公共客户端现支持 `--session-id` 和 `--previous-request-id`，用于从新会话开始的连续工作，不自动承接缺少sequential-v1回执的旧请求。每轮使用独立的 `--state` 文件，输入已知时可在上一轮queued/running期间提交后轮；依赖其输出时先等到succeeded再构造输入。客户端在新POST前读取前序并核对协议、会话和可受理状态，服务端最终原子核对唯一后继。下面三轮不依赖Happy或DSH；示例ID须替换为实际工作持久保存的ID，模型须从实时目录选择。先运行第一轮：

```sh
python3 examples/image_client.py --env-file .image-client.env chat-submit \
  --state ./work-1.json --request-id work-example-1 --session-id work-example \
  --model DISCOVERED_TEXT_MODEL --prompt '先给出商品文案的事实核对步骤。'
python3 examples/image_client.py --env-file .image-client.env chat-status --state ./work-1.json
```

第一轮成功后，只提供新增事实：

```sh
python3 examples/image_client.py --env-file .image-client.env chat-submit \
  --state ./work-2.json --request-id work-example-2 --session-id work-example \
  --previous-request-id work-example-1 --model DISCOVERED_TEXT_MODEL \
  --prompt '已核实商品为蓝色发夹，包装内2个，请继续核对。'
python3 examples/image_client.py --env-file .image-client.env chat-status --state ./work-2.json
```

第二轮成功后，沿同一会话继续：

```sh
python3 examples/image_client.py --env-file .image-client.env chat-submit \
  --state ./work-3.json --request-id work-example-3 --session-id work-example \
  --previous-request-id work-example-2 --model DISCOVERED_TEXT_MODEL \
  --prompt '请按已核实事实给出一句中文描述，不增加未证实属性。'
python3 examples/image_client.py --env-file .image-client.env chat-status --state ./work-3.json
```

收到202、断线或进程退出后，以该轮原状态文件运行 `chat-status`；需要有界读取上游时运行 `chat-recover`。相同 `chat-submit` 重入也只查该轮原ID，不重发。`unknown`、`failed`、会话或协议不符时不能提交后轮；服务器拒绝字段或回执不确认sequential-v1时也不能去掉会话字段重试。客户端保存会话/前序身份且将其纳入原输入指纹，恢复时核对原回执，不把最新一轮结果当旧轮结果。保存状态文件不是保存全部输入；应用仍应保留原提示词/附件，服务器正式受理后的恢复依赖其持久输入，不能用历史缺输入请求冒充新协议验收。

应用工具结果须在实际执行后作为下一轮新增user文本提交；这不是服务端工具调用协议，不发送 `role=tool` 或 `tools`。同一步的已知输入由调用方在受理前组为有序messages，服务保存完整顺序；不能合并或改写已受理的不同请求。每轮最后一条消息必须是user。`client_conversation_id`、`previous_request_id`遵循原请求ID字符规则，缺省字段省略而非null。账号、上游会话和父消息位置由服务器保留；客户端不能用这些字段选账号，也不要把保存的会话数当执行占用。

服务端在模型发送前持久保存本轮最后一个 user 的实际父消息和整次上传的根父消息，二者在多段上下文时不同。新连续会话必须取得本请求唯一终态分支，不能用流断开或聊天的“最新回答”推进下一轮。显式承接旧已成功的原请求时，先重新读取证明原结果及游标；缺证明则保留等待（旧非准入执行模式明确失败），不重新建聊、不改旧收据。原 UNKNOWN 不自动迁移。新客户端需与本协议的 Provider 成对部署；无此协议的服务会拒绝字段，不能静默退回每轮新聊。

Happy 主循环依据原生 DSH 消息 `source.replayState.response.requestId` 承接，并保留收到的消息指纹和系统/工具版本指纹。截图内多个连续 user 气泡可能是一次请求中的上下文数组，不应据此判断有几次上游发送。图片工具仍有独立持久图片任务；主推理的会话连续不等于把独立图片任务合并成一条图片请求。本增量不改变额度、并发设置、员工权限或原生 Codex 路线。


### Advanced account selection (separate from default setup)

Automatic account routing remains the default. For an advanced ordinary Chat text or image-input request, an application may copy an opaque `account_ref` from a text model's public `accounts` directory and include it in `POST /api/chat-requests`:

```json
{
  "client_request_id": "selected-chat-1",
  "model": "gpt-5-6-thinking",
  "account_ref": "car_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "messages": [{"role": "user", "content": "Explain the supplied product."}]
}
```

The example reference is a placeholder; use an actual advertised `car_` reference (43 URL-safe characters after the prefix). Never send an upstream token, email, or raw account identity. The companion client supports `chat-submit --account-ref <reference>` and saves this choice with its original request envelope. The default copyable instructions continue to use automatic routing.

A selection means **only that account**. The durable admission path queues a known account under the same request ID while it is unavailable, rate limited, or not currently observed to support the model; it never substitutes another account. It becomes executable only when that exact account and model are eligible. Deployments using the legacy executor without admission safely fail an unavailable selection instead of falling back. Invalid references fail schema validation (422); nonexistent or ambiguous references are rejected with `CHAT_ACCOUNT_NOT_FOUND` or `CHAT_ACCOUNT_AMBIGUOUS` (409).

Changing or removing an explicit selection on the same request ID is `CHAT_REQUEST_CONFLICT`. Retrying an identical ID reads the original receipt before account/model rediscovery, including after a restart or directory outage. Do not create a replacement ID to bypass an unknown original outcome. A sequential continuation inherits its predecessor's bound account when the selector is omitted; an explicit different account returns `CHAT_ACCOUNT_SELECTION_CONFLICT`. Original owner, session, recovery and archive rules continue to apply.

The image task JSON and multipart APIs also accept `account_ref`; the same durable choice reaches the original scheduler and final send. `submit --account-ref <reference>` supports both generation and edits. It is included in the local input fingerprint before POST, and changing or omitting it on a saved state is rejected. Keep the original ID during quota/observation waits. Image thread continuations must keep their original account; selecting another is an explicit conflict. Public receipt projections do not expose internal account identities, bindings, or credentials.

| Entrance | Selection and original identity |
| --- | --- |
| `/api/chat-requests` | JSON `account_ref`; persist `client_request_id`. |
| `/api/image-tasks/generations`, `/edits` | JSON or multipart `account_ref`; persist `client_task_id`. |
| `/v1/chat/completions`, `/responses`, `/messages`, `/search` | Optional safe reference, with pre-saved `X-Client-Request-ID`; preserved in durable admission and stripped before upstream protocol forwarding. |
| `/v1/images/generations`, `/edits` | Same safe reference; use the required original task identity and stable request header for selected compatibility calls. |
| `/codex/v1/responses`, `/responses/compact` | Safe reference plus stable request header; original native session/response affinity still wins and conflicting selection is rejected. |
| Company forwarding | The native Chat/image task entrances above retain the same JSON/multipart bytes. Company ingress does not open arbitrary compatibility or account-management routes. |

Malformed or null selection is rejected. Selected compatibility requests without a stable request header or durable admission fail explicitly, never fall back to an automatic account. Codex image aliases currently lack verified image-specific capacity evidence: they retain an original waiting receipt instead of borrowing text quota or probing by generating an image. This does not imply a working public persistent Codex image API.

For example, after `models` returns a real image account reference:

```sh
# Replace the placeholder with the opaque reference actually returned by models.
python3 examples/image_client.py --env-file .image-client.env submit \
  --state ./selected-image.json --client-task-id selected-image-1 \
  --model gpt-image-2 --prompt 'Create the requested product illustration.' \
  --account-ref DISCOVERED_ACCOUNT_REF
python3 examples/image_client.py --env-file .image-client.env status --state ./selected-image.json
```

The submit response can remain `queued` with a waiting reason. Exit and run `status` with the same state until the original receipt is successful, then use `download`; this example does not require Happy, DSH, or prior project conversations. Editing adds one or more `--image` arguments to the same flow. The caller never supplies private provider binding/account IDs.


## Complete a work task and return to its original conversation

A request result is one turn, not the end of the application's whole work. Keep the application work reference, original session/thread and last successful request ID together. After the application's review is complete and the final results have been saved, its work-complete action should archive by default. Do not archive a pending/unknown turn or clear a failed receipt to claim completion.

The same client supports `chat-complete --state ./work-3.json` for the last successful sequential Chat turn. On rework, run `chat-rework --state ./work-3.json`, confirm `archived:false`, then create a new turn with the same `--session-id` and `--previous-request-id work-example-3`. These commands call the existing `archive-conversation`/`restore-conversation` endpoints and store the original-work lifecycle result before/after the request. Network ambiguity remains `unknown`; explicit recovery uses the same original object, whose server-side method reads the actual state before any PATCH. It does not generate a new answer.

For a continuous image work, start `submit --thread-id product-work` with model `gpt-image-2`. Download the successful original image. An edit uses a new task/state, the same `--thread-id`, `--source-task-id ORIGINAL_SUCCESSFUL_TASK`, and `--image ./downloaded-original.png`; `--source-index` identifies that original among multiple supplied images (default 0). The server checks exact original-result bytes, owner and completed predecessor. Do not replace those with an upstream URL or another task's latest image.

After review and saving, run `complete --state ./last-image.json`; it calls `archive-thread` for that work's latest successful task. To rework, run `rework` on the same state, confirm the original thread is restored, then submit a new edit along that thread. Lifecycle operations reject incomplete, mismatched or superseded work, and keep all original results and immutable input identities.

The `--timeout` option is a network-wait timeout only. It does not cancel durable server work or release an unknown upstream outcome. Set a finite application in-flight count and query according to server backoff; stop local waiting by saving the original state and resume querying later. Do not invent unsupported server cancellation/deadline fields. Same-session turns are serialized, while independent sessions can share eligible account capacity under the existing scheduler and upstream pacing.

### Persisted scheduling and automatic work completion

Pass an optional `scheduling` object on model submission. Multipart edits encode it as JSON text in the `scheduling` form field. The server stores it with immutable input and applies it both in the original queue and immediately before sending:

| Field | Scope and default |
| --- | --- |
| `workflow_id` | Nonempty application workflow reference, at most 128 characters; grouped under the authenticated stable caller, not caller-supplied identity. |
| `workflow_concurrency` | 1–64 simultaneously held work-conversation slots; requires a workflow ID, whose default concurrency is 1. These are interface bounds, not measured safe account concurrency. |
| `min_send_interval_seconds` | Extra minimum interval between actual sends in this workflow, 0–86400; omitted adds no delay. Existing upstream pacing still applies. |
| `not_before` | UTC ISO8601 earliest actual send; omitted has no extra earliest time. |
| `wait_deadline` | UTC ISO8601 cutoff for still-unsent work; omitted has no extra cutoff. Already-sent or UNKNOWN work is never replayed or cancelled by this date. |

The response returns effective `scheduling` and scoped `waiting.reasons`. Same-work continuation inherits workflow/interval choices when omitted; timestamps apply only to their original request. A model turn ending does not free a continuing work's slot. A completed or paused work does. Distinct keys of one known caller do not gain extra scheduling shares; original receipt ownership is unchanged.

The application persists final results first (`chat-save --state ./work-3.json --output ./result.json`, or image `download`), then emits its work-complete event automatically:

```text
POST /api/chat-requests/ORIGINAL_LAST_ID/work
{"state":"completed","results_saved":true}

POST /api/image-tasks/ORIGINAL_LAST_ID/work
{"state":"completed","results_saved":true}
```

GET the same `/work` path to read `protocol:work-v1`, `kind`, `request_id`, `work_ref`, `state`, `slot_held`, and `archive`. Completion and the archive intent are stored atomically; `state:completed` releases this work's slot, while `archive.status:pending/running/unknown` truthfully retains unconfirmed upstream archive. Background recovery continues the original target after failure/restart; only `confirmed` with `archived:true` proves archive. `chat-work-status`/`work-status` expose this through the same standalone client. Existing `complete`/`rework` commands use compatible archive/restore entrances backed by the same durable work state; a timeout requires original work readback, not a new model submission.

POST `{"state":"paused"}` to suspend an unsent/finished-turn work without archiving. POST `{"state":"active"}` to resume or rework; an archived work remains `restoring` until original restore is confirmed. A late prior completion cannot close a new work or its slot. UNKNOWN/in-flight work rejects release. Legacy single-request protocols and native Codex without a verified upstream archive API explicitly use `archive.scope:provider_work` / `status:not_applicable`; no upstream archive is claimed.

For example, start two independent sessions under `--workflow-id product-copy --workflow-concurrency 2 --min-send-interval-seconds 12`. Each session keeps its own state files and ordered predecessor IDs. Add `--not-before`/`--wait-deadline` with actual UTC times when needed. Leave the same original ID in place while queued; the scheduler resumes it after capacity or timing constraints clear. `WAIT_DEADLINE_EXCEEDED` with `not_sent` ends only unsent waiting. The client's HTTP timeout is still separate from these server controls.
# Bounded completion of an abnormal pure-generation step

Native sequential Chat text automatically retries a confirmed empty reply **once,
in the same account and conversation, with the retained original input**. No
caller retry action or root-cause diagnosis is required first. The original
request must have a matching upstream user message and an empty text final;
normal SSE completion plus local execution completion establishes the response
boundary. Legacy receipts without stream diagnostics additionally require the
persisted bounded unchanged-result observation. A stale `in_progress` final does
not by itself block this path. A timeout, arbitrary EOF, missing message, media,
transport error or HTTP 200 alone is not confirmed empty. Known insufficient
quota still excludes the account before admission and at the send check.

The service reuses the durable completion record and creates one linked request.
It keeps the model, account_ref, effective scheduling and work slot, transfers
only this original's local turn reservation, and rechecks the actual conversation
head immediately before POST. A completed manual continuation is preserved and
the retry appends after its latest head; changed or unreadable evidence keeps the
attempt unsent. This is not a remote cancellation claim: `original_turn_ended`
may remain false, the original receipt stays UNKNOWN, and `stop.confirmed` stays
false. `local_reservation=transferred/released` reports local accounting explicitly.

GET of the original request exposes `completion` progress, `replacement_id`,
`selected_id` and the selected result without rewriting the original receipt.
The supplied CLI's normal `chat-save` (images: `download`) saves that selected result and retains both
IDs. A second empty reply stops automatic sends (`COMPLETION_ATTEMPT_EXHAUSTED`)
and releases that verified-empty attempt's local account turn while preserving
UNKNOWN and conversation ordering for diagnosis; no silent prompt shortening, new account,
new conversation or unbounded retry is performed. Business output requirements
must still be checked before work completion and archive. Explicitly pausing a
verified-empty test work releases its local turn and workflow slot without
claiming remote cancellation or discarding the failed sample. Unconfirmed
transport failures and in-flight requests do not get this exception.

For other approved pure-generation recovery, POST to the original
`/api/chat-requests/{original-id}/completion` (images: `image-tasks`) with
`{"action":"recover"}`. Strict ended-empty evidence continues to support the
existing same-conversation path. Uncertain results that do not meet the empty
response rule return `COMPLETION_ORIGINAL_END_UNCONFIRMED`. The current adapter
has no verified remote stop operation; no local disconnect or archive pretends
to stop upstream generation. Image `/resume-poll` remains result recovery only.

Newly accepted native Chat text and retained Chat image generation requests also
use bounded failure recovery. An explicitly unsubmitted transient failure is
requeued once under the **same request ID and exact stored input**. Long-stalled
sent attempts get a 15-minute no-progress window plus a bounded five-minute
investigation. Only a fresh, unbranched original user-message chain with no usable
result permits one linked retry in the **original account and conversation**.
The current head is rechecked immediately before POST. Partial text, generated
image references, a later user turn, missing history, failed reads or unknown
bindings block this path; the service does not reconstruct another conversation.
Known image results only resume download. The retry keeps the original prompt,
model, account selection, scheduling and image inputs; it never silently shortens
or rewrites a task. Historical UNKNOWN receipts are not opted in by an upgrade.
The existing `allow_unconfirmed_retry:true` flag explicitly enables this same
bounded original-conversation policy for an older pure-generation request; it
no longer authorizes a new conversation.

When a bounded attempt is retired, automatic reads of that attempt stop and its
local turn is released or transferred to its unique child. After a second failed
attempt, `COMPLETION_ATTEMPT_EXHAUSTED` ends automatic sending and releases the
work slot only if no other member is executing, reading or pending. A query
failure keeps its real error and UNKNOWN evidence; it is never rewritten as
“not sent.” `execution.attempt_state=ended` and `execution.result_state` distinguish
this local ending from an upstream terminal result. Other operator pauses remain
in force. A successful child still requires actual save/review before completion,
archive and rework of that original conversation. No remote cancellation or
exactly-once guarantee is implied. These rules exclude external business writes,
compatibility forwarding and Codex routes.

The independent client accepts explicit authenticated `recovery.upstream_outcome`
or `detail.upstream_outcome=not_sent` as transport evidence. A transient 429/503
with Retry-After permits one automatic same-ID transport retry; long cooldowns
are persisted for a later invocation. A subsequent invocation may resend only
when this proof is retained, input is unchanged and the exact original ID is
absent. HTTP 404/429 alone, a timeout or a client crash never proves non-submission.
The client saves `phase=unknown` before each POST and replaces that state only
with an actual response; it never clears the original ID.

`GET` the same `/completion` to inspect `state`, `reason`, `waiting`,
`original_status`, `replacement_status`, `replacement_id` and `selected_id`.
Selection is durable and write-once. An original found before the replacement
send cancels the unsent replacement. Once the replacement has been sent, its
completion retains result ownership; a late original cannot strand it. GET and upstream POST are not atomic: a
late original can still appear after replacement submission. Both receipts stay
available; only the selected result enters downstream saving/review.

The shipped independent CLI preserves its original state file:

```sh
python examples/image_client.py chat-completion-recover --state original.json
python examples/image_client.py chat-completion-status --state original.json
python examples/image_client.py chat-completion-save --state original.json --output answer.json
python examples/image_client.py chat-completion-complete --state original.json --reviewed
python examples/image_client.py chat-completion-rework --state original.json
```

For images omit the `chat-` prefix and save an image output. `complete` verifies
the selected local file still matches the saved bytes, then acknowledges the
result as saved and reviewed. `result_ready` is not completion. API clients send
`{"action":"complete","selected_id":"returned-id","results_saved":true,"reviewed":true}`
only after their actual durable save/readback/review. The selected conversation
archives at that task-complete boundary. A same-conversation correction closes
its shared work only after the verified correction has succeeded and its result
is selected; an ended-empty original alone cannot be marked completed. `work.archive` reports pending, failure
or confirmation; `original_work.cleanup_pending` separately preserves old
unknown occupancy and its deferred cleanup. Rework restores the selected
conversation and never sends a new model request. The original conversation
is retained. Applications continue using the selected result's public session
reference and ID after confirmed restore. Historical UNKNOWN receipts do not
opt in merely because the service upgrades; new pure-generation requests use
the bounded policy described above.
