# Persistent image-task client

## 中文接入要点

第二阶段双端权限已于2026-09-17部署并由用户确认。图片/文字程序勾选「允许 Chat」，需要 Codex 时可另勾选「允许 Codex」；同端操作没有用途子权限。修改允许端不删除原回执；撤销后认证失效。发布时两把保留旧key均双端，三把旧key保持撤销，新创建密钥以其当前政策为准。

首选入口是工作台「ChatGPT / Codex → 密钥与 CLI → 复制给AI配置」。复制说明不包含真实密钥，交给负责目标应用的AI；密钥通过本地私密环境配置另行提供。公开同源合同位于 https://app.hugsweetglobal.com/ai/integration.md ，不需要网站后台登录。下文为已有Python客户端的备用操作说明。第三阶段公共Chat增量的真实部署/接入证据单列，不以第二阶段发布推断第三阶段已经验收。

- `SERVER_ROOT` 是服务根（`https://app.hugsweetglobal.com/ai`）；兼容 API Base URL 另加 `/v1`。持久任务不能拼成 `/v1/api/...`。
- 在工作台“AI 服务连接”中创建普通调用密钥，仅通过私有渠道交付，填入本地权限为 `600` 的环境文件。它没有账号池管理员能力；撤销后原密钥立即失效。
- 用下面的 `submit` 保存原任务号，再用 `status` 查询、`download` 保存图片。编辑时加 `--image 本地图片`，可传多张参考图。
- 进程退出、网络超时或结果未知后，继续使用原状态文件和任务号。客户端不会自动换号或重新生成。
- 可靠共用候选完整受理后先持久保存实际输入；池满返回原 `queued` 回执，新增/恢复账号后服务自动派发，服务重启也不要求重提。`waiting` 描述当前原因。部署前后的行为以实际版本为准，见[部署说明](deployment.md#可靠共用与动态容量候选2026-09-20)。
- 账号界面只管理自己导入的账号，显示上游观测值与更新时间；未知、真实零、读取失败和调用后待刷新分开。没有内部预算或张数分配。
- 同源工作台界面无需 CORS。跨域浏览器调用默认关闭；管理员需将明确来源写入 `CHATGPT2API_CORS_ORIGINS`（逗号分隔）或配置 `cors_origins`，不开放凭据通配来源。普通后台 HTTP 程序不受 CORS 影响。
- 本接入包不等于小乐的程序已接通。她的调用位置确定后，才能核对 SDK、DSH 或其它调用方式的最小适配。

`examples/image_client.py` is a dependency-free Python client for an external caller that needs a durable image-task receipt. It uses only the Python standard library. Its local state contains no bearer token, prompt text, or image bytes.

This is an engineering client for the persistent task API. The existing public service root is `https://app.hugsweetglobal.com/ai`. Local fixture tests do not establish a reachable endpoint, a usable provider account, or a successful external image generation.

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

Execution currently reuses the service's existing paid-account conversation-binding route. Free accounts are not eligible for this route. A response saying that no eligible image resource was admitted describes the route at that moment; it is not evidence that every account in the private pool has exhausted an upstream quota.

The synchronous-compatible routes use the same persistent receipt machinery as `/api/image-tasks`. Supply a stable `client_task_id` on every synchronous request. A completed request returns HTTP 200 with `b64_json` data and the original task ID. Queued/running requests or requests awaiting authoritative recovery return HTTP 202 with that same ID. In the reliable-sharing candidate, accepted requests wait durably for account capacity; an older retained resource-admission failure may still return HTTP 429, and other known terminal failures return HTTP 502. Transport/body-reader 429 before acceptance is distinct from upstream 429: inspect `rate_limit.layer`, phase, request ID and Retry-After. Never create a new ID automatically after timeout or non-200. Poll original receipts; generated-but-undownloaded results resume downloads only.

## Commands

The client reads the env file only when `--env-file` is supplied. Global options precede the command.

Read the service model catalog before submitting. For image output select `gpt-image-2`; text-capable entries have `capabilities` including `text` and `image_input`:

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
- `429`: no currently eligible provider resource was admitted; this does not by itself prove a global upstream quota value;
- `5xx`: the server or its upstream path failed.

An HTTP response is recorded in the local state. A timeout, disconnect, invalid response, or interruption after the state was prepared has an unknown submission outcome. The client never converts that uncertainty into a new ID and never silently retries the submit POST. Restart with the same state and run `status`, or run the identical `submit` command, which performs the same status lookup. If the server reports that the original ID is missing, the client stops; deciding to create a different task requires an explicit new state file and ID.

Do not put the bearer token in command arguments, task state, logs, screenshots, or source control. Keep the filled env file private.


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

技术保护按服务进程限制为2个请求体读取、32个执行或等待的文字任务、256MiB保留输入内存，不是密钥预算。429 `CHAT_BODY_READER_CAPACITY_EXCEEDED` 表示读取繁忙。队列满会保存原ID回执：`failed` / `TEXT_TASK_CAPACITY_EXCEEDED` / `recovery.upstream_outcome=not_sent`；只有这项明确未发送证据允许应用退避后原ID原输入再次POST。示例客户端保持查询优先，不自动执行这个重试，也不能删除状态文件换ID绕过保护。

Chat成功状态是 `succeeded` 且有content；图片任务成功状态仍为 `success`，不要混用。只有实际取得的上游usage才可报告，当前未返回usage时应显示未知。现有公共Chat不提供工具执行或结构化输出保证；应用自行校验模型文本，不把JSON解析失败包装为成功。


### 连续工作会话（sequential-v1 增量）

持续推进的应用可在现有 `/api/chat-requests` 请求中提供 `client_conversation_id`；这是调用者的工作会话引用，不是上游 ChatGPT conversation ID。每轮使用不同的 `client_request_id`，除首轮外传 `previous_request_id` 指向已成功的原请求。每次只提供新增输入/工具结果与必要附件，不把全部旧历史再次追加。服务在原 owner 范围和原 SQLite 插入事务中核对前序、唯一后继，并沿前轮实际账号、上游会话和已证明最终回答承接；不允许客户端传管理绑定或原始上游游标。

回执增加 `conversation={protocol:"sequential-v1",client_conversation_id,previous_request_id}`，不暴露池账号/上游ID。前序待定返回409 `CHAT_PREVIOUS_REQUEST_PENDING`；漏前序、跨会话或已有后继返回明确409；这些拒绝不代表新请求已受理。查询原前序，不换新ID绕过。客户端确认原前序 succeeded 后才提交后轮。重复同ID同输入仍返回原回执，即使会话已推进到更后面；旧不带此字段的请求保持原哈希及单次调用方式。

服务端在模型发送前持久保存本轮最后一个 user 的实际父消息和整次上传的根父消息，二者在多段上下文时不同。新连续会话必须取得本请求唯一终态分支，不能用流断开或聊天的“最新回答”推进下一轮。显式承接旧已成功的原请求时，先重新读取证明原结果及游标；缺证明则保留等待（旧非准入执行模式明确失败），不重新建聊、不改旧收据。原 UNKNOWN 不自动迁移。新客户端需与本协议的 Provider 成对部署；无此协议的服务会拒绝字段，不能静默退回每轮新聊。

Happy 主循环依据原生 DSH 消息 `source.replayState.response.requestId` 承接，并保留收到的消息指纹和系统/工具版本指纹。截图内多个连续 user 气泡可能是一次请求中的上下文数组，不应据此判断有几次上游发送。图片工具仍有独立持久图片任务；主推理的会话连续不等于把独立图片任务合并成一条图片请求。本增量不改变额度、并发设置、员工权限或原生 Codex 路线。
