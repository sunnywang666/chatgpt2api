# Persistent image-task client

## 中文接入要点

第二阶段候选增加普通密钥用途：图片程序创建密钥时勾选「Chat图片生成」；Codex编程须另勾选。修改用途只影响后续提交，原任务查询、下载、原回执继续读取仍按原所有者校验；撤销后全部调用失效。缺少用途返回 403 `KEY_CAPABILITY_DENIED`，不要自动新建任务重试。旧密钥切换待逐把核对，尚未生产发布，详见 [部署指南](deployment.md#工作台程序密钥政策第二阶段)。

目前是待发布候选，下面的正式域名路径尚未完成外网和真实生图验收。部署核验完成后再交调用方使用。

- `SERVER_ROOT` 是服务根（计划为 `https://app.hugsweetglobal.com/ai`）；兼容 API Base URL 另加 `/v1`。持久任务不能拼成 `/v1/api/...`。
- 在工作台“AI 服务连接”中创建普通调用密钥，仅通过私有渠道交付，填入本地权限为 `600` 的环境文件。它没有账号池管理员能力；撤销后原密钥立即失效。
- 用下面的 `submit` 保存原任务号，再用 `status` 查询、`download` 保存图片。编辑时加 `--image 本地图片`，可传多张参考图。
- 进程退出、网络超时或结果未知后，继续使用原状态文件和任务号。客户端不会自动换号或重新生成。
- 账号界面只管理自己导入的账号，显示上游观测值与更新时间；未知、真实零、读取失败和调用后待刷新分开。没有内部预算或张数分配。
- 同源工作台界面无需 CORS。跨域浏览器调用默认关闭；管理员需将明确来源写入 `CHATGPT2API_CORS_ORIGINS`（逗号分隔）或配置 `cors_origins`，不开放凭据通配来源。普通后台 HTTP 程序不受 CORS 影响。
- 本接入包不等于小乐的程序已接通。她的调用位置确定后，才能核对 SDK、DSH 或其它调用方式的最小适配。

`examples/image_client.py` is a dependency-free Python client for an external caller that needs a durable image-task receipt. It uses only the Python standard library. Its local state contains no bearer token, prompt text, or image bytes.

This is an engineering client for the persistent task API. The planned public root is `https://app.hugsweetglobal.com/ai`, but it has not been deployed or live-validated. Local fixture tests do not establish a reachable endpoint, a usable provider account, or a successful external image generation.

## Server requirements

Set `SERVER_ROOT` to the service root used by the external caller. The root may include an ingress prefix, but it must not end in `/v1`. The example environment contains the planned `/ai` address and labels it as unvalidated. Do not reuse an OpenAI SDK `OPENAI_BASE_URL` value ending in `/v1`: doing so would incorrectly produce `/v1/api/...` paths.

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

The synchronous-compatible routes use the same persistent receipt machinery as `/api/image-tasks`; they are not a separate direct-generation path. Supply a stable `client_task_id` on every synchronous request. A completed request returns HTTP 200 with `b64_json` data and the original task ID. A request still running or awaiting authoritative recovery returns HTTP 202 with that same ID so the caller can query the persistent task. A known resource-admission failure returns HTTP 429, and other known terminal failures return HTTP 502, with the retained task ID in either case. Do not automatically create a new ID after any timeout or non-200 response.

## Commands

The client reads the env file only when `--env-file` is supplied. Global options precede the command.

Read the service model catalog before submitting. The external contract currently permits only `gpt-image-2`; discovery still confirms whether the deployed service actually advertises it:

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
