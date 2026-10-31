<h1 align="center">ChatGPT2API</h1>


<p align="center">ChatGPT2API 主要是对 ChatGPT 官网相关能力进行逆向整理与封装，提供面向 ChatGPT 图片生成、图片编辑、多图组图编辑场景的 OpenAI 兼容图片 API / 代理，并集成在线画图、号池管理、多种账号导入方式与 Docker 自托管部署能力。</p>

> [!WARNING]
> 免责声明：
>
> 本项目涉及对 ChatGPT 官网文本生成、图片生成与图片编辑等相关接口的逆向研究，仅供个人学习、技术研究与非商业性技术交流使用。
>
> - 严禁将本项目用于任何商业用途、盈利性使用、批量操作、自动化滥用或规模化调用。
> - 严禁将本项目用于破坏市场秩序、恶意竞争、套利倒卖、二次售卖相关服务，以及任何违反 OpenAI 服务条款或当地法律法规的行为。
> - 严禁将本项目用于生成、传播或协助生成违法、暴力、色情、未成年人相关内容，或用于诈骗、欺诈、骚扰等非法或不当用途。
> - 使用者应自行承担全部风险，包括但不限于账号被限制、临时封禁或永久封禁以及因违规使用等所导致的法律责任。
> - 使用本项目即视为你已充分理解并同意本免责声明全部内容；如因滥用、违规或违法使用造成任何后果，均由使用者自行承担。
> - 本项目基于对 ChatGPT 官网相关能力的逆向研究实现，存在账号受限、临时封禁或永久封禁的风险。请勿使用你自己的重要账号、常用账号或高价值账号进行测试。


## 赞助商

<table>
  <tr>
    <td width="190" align="center">
      <a href="https://www.atlascloud.ai/zh?utm_source=github&utm_medium=link&utm_campaign=chatgpt2api"><img src="assets/atlascloud.svg" width="163" alt="Atlas Cloud"></a>
    </td>
    <td>
      <a href="https://www.atlascloud.ai/zh?utm_source=github&utm_medium=link&utm_campaign=chatgpt2api">Atlas Cloud</a> is a full-modal AI inference platform that gives developers a single AI API to access video generation, image generation, and LLM APIs. Instead of managing multiple vendor integrations, you connect once and get unified access to 300+ curated models across all modalities. Check out <a href="https://www.atlascloud.ai/console/coding-plan">Atlas Cloud's new coding plan promotion</a> for more budget-friendly API access.
    </td>
  </tr>
</table>

## 快速开始

现有 Workbench 的普通密钥接入、持久任务与可恢复 Python 示例见 [外部生图客户端说明](docs/external-image-client.md)。部署与真实调用状态以该说明的核验记录为准。

### Docker 运行

```bash
git clone git@github.com:basketikun/chatgpt2api.git
cd chatgpt2api
docker compose up -d
```

启动前请先在 `config.json` 中设置 `auth-key`，也可以在 `docker-compose.yml` 中通过 `CHATGPT2API_AUTH_KEY` 覆盖。

- Web 面板：`http://localhost:3000`
- API 地址：`http://localhost:3000/v1`
- 数据目录：`./data`

### WARP / FlareSolverr 稳定代理部署

如果图片链路经常遇到 Cloudflare 拦截，可以启用附带的 WARP + Privoxy + FlareSolverr 方案：

```bash
cp .env.example .env
docker compose -f docker-compose.warp.yml up -d --build
```

该 compose 会启动：

- `warp-proxy`：提供 WARP SOCKS5 出口。
- `privoxy`：把 WARP SOCKS5 转成 HTTP 代理。
- `flaresolverr`：刷新 Cloudflare clearance。
- `init-config`：幂等写入 `proxy_runtime` 默认配置。
- `app`：启动 ChatGPT2API 主服务。

默认只让上游 OpenAI / ChatGPT 请求走稳定代理，账号邮箱、CPA 等辅助链路不会被强制接管。账号自身配置的代理优先级最高，其次是稳定代理运行时，再其次是显式代理和旧版全局代理。

可在 `.env` 中调整端口和代理运行时参数，也可在后台设置页的「稳定代理运行时」面板手动保存、测试代理和测试 clearance。

### 本地开发

启动后端：

```bash
git clone git@github.com:basketikun/chatgpt2api.git
cd chatgpt2api
uv sync
uv run main.py
```

启动前端：

```bash
cd chatgpt2api/web
bun install
bun run dev
```

后续更新新版本：

```bash
docker pull ghcr.io/basketikun/chatgpt2api:latest
docker-compose down
docker-compose up -d

```

### 内部 CI 队列回退

受信任分支的 POOL-02 CI 和 Docker 发布默认使用公司 runner。一个独立的 GitHub-hosted observer 只会在原 run 已排队 90 秒、所有 job 都未领取且确认取消后，重放同一受控 source SHA 到 GitHub-hosted runner。已开始、已结束、来源分支已前移、workflow 定义不一致或取消状态不确定时不会回退。observer 不 checkout 或执行候选 Provider 源码；它只读取默认分支中的受控 helper，且不读取或保存额外凭据。为保留 tag push 的版本、latest 与多架构发布语义，tag 发布不自动回退；只有既有 `workflow_dispatch` publisher 合同可自动重放。达到 GitHub Actions 预算上限时 observer 可能无法启动，但不会阻断默认公司 runner 路径。

### 存储后端配置

支持通过环境变量 `STORAGE_BACKEND` 切换存储方式：

- `json` - 本地 JSON 文件（默认）
- `sqlite` - 本地 SQLite 数据库
- `postgres` - 外部 PostgreSQL（需配置 `DATABASE_URL`）
- `git` - Git 私有仓库（需配置 `GIT_REPO_URL` 和 `GIT_TOKEN`）

示例：使用 PostgreSQL

```yaml
environment:
  - STORAGE_BACKEND=postgres
  - DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## 功能

### API 兼容能力

- 兼容 `POST /v1/images/generations` 图片生成接口
- 兼容 `POST /v1/images/edits` 图片编辑接口
- 兼容面向图片场景的 `POST /v1/chat/completions`
- 兼容面向图片场景的 `POST /v1/responses`
- `GET /v1/models` 返回 `gpt-image-2`、`codex-gpt-image-2`、`auto`、`gpt-5`、`gpt-5-1`、`gpt-5-2`、`gpt-5-3`、`gpt-5-3-mini`、
  `gpt-5-mini`
- 支持通过 `n` 返回多张生成结果
- 支持生成可编辑 PPT 文件
- 支持生成可编辑 PSD 文件
- 支持 Codex 中的画图接口逆向，仅 `Plus` / `Team` / `Pro` 订阅可用，模型别名为 `codex-gpt-image-2`，如有需要可自行在其他场景映射回
  `gpt-image-2`，用于和官网画图区分；也就意味着同一账号会同时有官网和 Codex 两份生图额度

### 在线画图功能

- 内置在线画图工作台，支持生成、图片编辑与多图组图编辑
- 支持 `gpt-image-2`、`codex-gpt-image-2`、`auto`、`gpt-5`、`gpt-5-1`、`gpt-5-2`、`gpt-5-3`、`gpt-5-3-mini`、`gpt-5-mini` 模型选择
- 编辑模式支持参考图上传
- 前端支持多图生成交互
- 本地保存图片会话历史，支持回看、删除和清空
- 支持服务端缓存图片URL
- 图片生成进度追踪；绑定 Chat 的任务取得账号容量和会话执行权后，保存本轮 300 秒执行期限，重启不会重新计时
- 到期查询原请求：仍在生成则保留占用；连续三次有效读取确认没有活动生成或可用结果后，返回可恢复的 `RESULT_UNRECOVERABLE`。网络或限流查询失败不算“没有结果”，并遵守上游冷却时间
- 已取得输出图片的任务单独恢复下载，不重新生图；查询结果携带安全的故障类别、恢复动作及下次检查时间，供调用方显示准确进度
- 图片懒加载与滚动位置记忆，优化大量图片场景性能

### 号池管理功能

- 自动刷新账号邮箱、类型、额度和恢复时间（异步进度追踪）
- 轮询可用账号执行图片生成与图片编辑
- 遇到 Token 失效类错误时自动剔除无效 Token
- 定时检查限流账号并自动刷新
- 支持密码重新登录恢复异常账号，刷新后可自动重登
- 支持网页端配置全局 HTTP / HTTPS / SOCKS5 / SOCKS5H 代理
- 支持 WARP / FlareSolverr 稳定代理运行时
- 支持搜索、筛选、批量刷新、导出、手动编辑和清理账号
- 支持四种导入方式：本地 CPA JSON 文件导入、远程 CPA 服务器导入、`sub2api` 服务器导入、`access_token` 导入
- 支持在设置页配置 `sub2api` 服务器，筛选并批量导入其中的 OpenAI OAuth 账号

### 实验性 / 规划中

- 详细状态说明见：[功能清单](./docs/feature-status.en.md)

## 效果展示

<table width="100%">
  <tr>
    <td width="50%"><img src="https://i.ibb.co/Jj8nfwwP/image.png" alt="image" border="0"></td>
    <td width="50%"><img src="https://i.ibb.co/pqf235v/image-edit.png" alt="image edit" border="0"></td>
  </tr>
  <tr>
    <td width="50%"><img src="https://i.ibb.co/tPcqtVfd/chery-studio.png" alt="chery studio" border="0"></td>
    <td width="50%"><img src="https://i.ibb.co/PsT9YHBV/account-pool.png" alt="account pool" border="0"></td>
  </tr>
  <tr>
    <td width="50%"><img src="https://i.ibb.co/rRWLG08q/new-api.png" alt="new api" border="0"></td>
  </tr>
</table>

### 绑定文本请求的异步恢复

`POST /api/conversation-bindings/text` 可携带调用方生成的 `client_request_id`。服务先在 `data/text_tasks.sqlite3` 保存请求记录，立即返回 `request_id` 和 `queued/running` 状态，再处理图片上传与对话；不携带该字段时保留原同步响应。

使用相同身份调用 `GET /api/conversation-bindings/text-requests/{request_id}` 查询进度、原账号和对话、完成结果。相同请求编号与相同输入只执行一次，输入改变返回冲突；查询不会重新生成。绑定文本和后续改图保留 Chat 对话历史，沿原账号与对话继续；不沿用普通临时文本请求的关闭历史设置。记录不保存访问令牌、请求提示词或上传图片字节。重启后结果不明的记录只按原用户消息编号查询，不能把上一轮回答当成本轮结果，也不自动重发未知请求。调用方对明确失败负责有限退避重试，成功阶段继续复用。

对已停止的 no-final 文本后继，`CHAT_SUPERSEDE_CURSOR_CHANGED` 可附带有限的 `no_final_guard` 说明，帮助界面显示本次保护性拒绝的原因。该字段只在失败回执仍是该错误时出现，格式为 `code`、`phase`、`reason`、`message`；例如：`{"code":"CHAT_SUPERSEDE_CURSOR_CHANGED","phase":"pre_send_no_final_tail","reason":"current_node","message":"Conversation cursor changed"}`。`phase` 仅为 `initial_no_final_tail`、`pre_send_no_final_tail` 或 `context_changed`；前两种允许有限状态原因，`context_changed` 仅为 `current_node`。说明不包含游标、消息内容、账号、异常原文或会话图；它解释本次失败，不会自动重新发送或放宽原有保护。已结束的旧回执如果当时没有保存该说明，读取时不会补猜原因。

携带 `product_work` 的文字任务在流结束后，使用原有的那次会话读取核实本次用户消息的唯一完成分支，正文和回复位置均取自该分支。后续对话推进不会改变原请求的结果；找不到原节点、分支不唯一或尚未完成时保留结果未知，继续按原请求查询，不重新生成。此核实不增加上游读取次数。

已保存成功文字结果却残留旧恢复占用时，普通查询会在同一事务内核实属于普通 Chat 文本请求、结果已完成且完成时间有效、执行与预留均已结束、无未完成或恢复错误状态、无旧式恢复占用字段、恢复租约及仍存在的执行租约已过期，再清理该恢复标记；此处理不查询上游、不重发，并保留原结果。有效或无法核实的租约、暂停、结果未知、转发及图片恢复不会按此规则清理。调用方仍须采用并保存结果，按原工作流程完成关闭与归档，不能将恢复标记清理当作整项工作完成。

### 统一只读任务结果

`GET /api/task-results/{kind}/{request_id}` 读取账号池已保存的结果，`kind` 为 `text` 或 `image`。请求身份必须与原任务所有者一致；不存在或属于其他所有者的任务返回 404。接口继续执行对应的 Chat、Codex 或图片能力权限检查。

```json
{
  "protocol": "task-result-v1",
  "kind": "text",
  "request_id": "original-request-id",
  "status": "succeeded",
  "result_ready": true,
  "result": {"content": "已保存的完整正文"},
  "receipt": {"request_id": "original-request-id", "status": "succeeded"}
}
```

外层状态统一为 `not_started`、`queued`、`running`、`succeeded`、`failed` 或 `unknown`；本地停止等待但上游结果未确认时仍为 `unknown`。文本结果为 `result.content`，图片结果为 `result.data`；未就绪时 `result` 为 `null`。`result_ready` 只表示已有非空文本或通过本地格式、存储引用检查的图片结果，不证明工作台已采用、图片下载成功或商品已发布。图片下载仍走返回的受保护地址。

`receipt` 保留既有回执语义，供调用方校验原请求、会话与接续条件。内部管理员调用保留工作台需要的绑定字段，普通调用及外部入口仍按既有规则脱敏。上面的 `receipt` 仅作最小示例；消费者不得用它替代实际身份与游标校验。

该 GET 直接读取持久化记录，不查询上游、不触发恢复、不重发、不清理占用，也不需要工作台通过轮询驱动任务执行。账号池的后台执行与原结果恢复负责保存结果和后续调度。恢复命令应显式调用既有恢复入口，不能由普通结果读取隐式触发；未知结果不授予重发权限。

已有 SSE 通知协议仍为 `task-notification-v1`，只提示结果可能就绪。工作台收到提示后通过上述只读接口核实，再幂等保存业务成果并推进下一步；通知丢失时可有界补读。旧查询接口保持兼容，其可能触发原结果恢复或清理的行为不属于本接口的只读保证。

### 工作台的结果读取、原结果补取与重新尝试

这三种操作必须由调用方明确区分。现有接口已经支持原结果补取，无需为了统一名称增加另一套恢复状态机：

| 意图 | 现有入口 | 边界 |
| --- | --- | --- |
| 读取已保存结果 | `GET /api/task-results/{kind}/{request_id}` | 纯读取；正常状态检查和通知后的核实使用它。 |
| 补取原文字答案 | `POST /api/conversation-bindings/text-requests/{request_id}/recover` | 按原 owner、请求和会话读取；受原暂停、租约与退避约束，不创建新生成请求。 |
| 补取原图片结果或继续下载 | `POST /api/image-tasks/{task_id}/resume-poll` | 继续原任务的查询或下载；不创建新生图请求。 |
| 读取保存的图片 | `GET /api/image-tasks/{task_id}/images/{index}` | 按原所有者和成功回执授权下载；不重新生图。 |

补取入口的 `allow_unrecoverable_retry` 默认为 `false`。它与 completion 的 `allow_unconfirmed_retry` 不是同一权限；不能仅因名称含 retry 就互相替换。已有明确允许重新检查原结果的分支应显式调用补取入口，普通结果读取不得暗中升级为补取。补取返回 HTTP 200 也可能仍为等待、延期或结果未知，调用方必须检查回执，不能据此推进为成功。

重新尝试目前有不同的适用范围，不能将一种入口的权限套用到另一种：

- 内部绑定文本通过 `POST /api/conversation-bindings/text` 的 `retry_unsent_request_id` 或 `supersedes_request_id` 引用原请求。调用方提供后继请求 ID 和既定绑定身份，Provider 从保存的原输入重建请求并检查 hash、原请求资格、活动占用、会话顺序和单后继；不允许凭普通 UNKNOWN 或仅 HTTP 超时重发。`retry_unsent_request_id` 是特定已证未发场景的新子请求，不等同于任意原 ID 重排。
- 普通客户端的 `POST /api/chat-requests/{id}/completion` 和 `POST /api/image-tasks/{id}/completion` 保持既有有界 completion 契约，公司入口保持既有原任务恢复及成果确认限制。内部绑定任务不得通过伪造普通 Chat receipt、改写 route 或未发送状态进入普通客户端策略。
- 工作台的内部管理员调用使用同一组 completion 路径，但必须明确区分 `action: "recover"` 与 `action: "retry"`：前者仅补取原请求，后者才请求账号池核验并执行受控的新尝试。新尝试不接受替换提示词、模型、绑定身份或调用方声称的未发送证明；账号池读取保存的原输入和真实回执，继续执行原有资格、会话顺序、单后继、暂停与发送前校验。普通 UNKNOWN、HTTP 超时及仅有旧查询游标均不能授权重发。内部调用不开放 `allow_unconfirmed_retry`。
- 文本 `ORIGINAL_RESULT_NO_FINAL` 继续使用既有的有限补读与后继资格策略，并在发送前核对原会话分支。这表示符合该策略的受控重试，不表示账号池已证明上游生成被取消；迟到成功或分支变化仍须阻止再次发送。
- 上述内部命令不改变工作台的业务重试预算。消费者应先接纯读和明确命令，验证相同请求的重复调用、重启及迟到结果，再删除对应重复编排；旧后继协议在兼容迁移完成前保留。

内部 `retry` 的请求体只接受 `{"action":"retry"}`，返回 `task-attempt-v1`：

```json
{
  "protocol": "task-attempt-v1",
  "kind": "text",
  "original_id": "original-request",
  "attempt_id": "provider-selected-attempt",
  "status": "queued",
  "result_ready": false,
  "existing": false,
  "created": true
}
```

`attempt_id` 是后续纯读的对象，可能为已成功的原请求、既有后继或本次安排的尝试。`created` 只描述本次是否安排尝试，不证明请求已经发给上游，更不证明业务成果已保存。`status` 复用 `task-result-v1` 的状态投影，未知结果不报作失败；最终可采用结果仍通过该纯读接口核实。

如果内部 `retry` 的响应丢失，可用 `GET /api/chat-requests/{original_id}/completion?action=retry`（图片为 `/api/image-tasks/{original_id}/completion?action=retry`）纯读找回已持久化的尝试。该查询仅向内部管理员开放，返回相同 `task-attempt-v1` 身份，固定 `existing:true`、`created:false`；不发送、不查询上游、不取消子请求或清理占用。文本按原 owner 及唯一合法 child 关联读取，已发或 UNKNOWN 的 child 不被迟到原答案覆盖；图片只查回已决定的同 ID 尝试，不能接管其他 replacement。未发现已决定尝试且原请求未成功时返回 `404 TASK_ATTEMPT_NOT_FOUND`，冲突返回 `409 TASK_ATTEMPT_CONFLICT`。404 不是未发送证明，不能因此重新 POST。调用方仍须持久保存原业务与 `attempt_id` 的关系，再通过 task-results 采用成果。不带 `action=retry` 的普通 completion GET 保持既有行为，不属于此纯读保证。

此入口的当前范围有意受限：文本复用既有 no-final 单后继策略，或严格的未发送绑定失败重试。后一种由 Provider 在原绑定会话内证明原请求节点不存在、旧游标至新游标为同一条无新用户消息的链，且新游标是当前已结束的末端；发送前再次核验。缺少这些证据时返回 `CHAT_BOUND_RETRY_CURSOR_UNAVAILABLE`，不能拿失败前的旧游标重放，也不能把前序回合的回答当成本请求结果。图片只将严格证实未发送的原任务按原 ID 重排，新增后继预算为零；该次发送以后若结果为 UNKNOWN，不允许再凭先前的重试命令创建第二次生成。新 `retry` 不授予普通或公司密钥。

工作台保留商品质量判断、业务重试预算、原业务 ID、成果采用与发布；账号池保留原输入、实际收发、账号与物理会话顺序、原结果保存及防重。商品属性或文案派生规则迁移不属于接口改名，需在消费者接齐后单独验证。

`category_directory_parent_v1`、`attributes_required_only_v4` 和 `combined_copy_compact_v1` 仍分别由 `services/category_directory_derivation.py`、`services/required_attributes_derivation.py`、`services/combined_copy_derivation.py` 提供旧调用的兼容实现。工作台持有迁移后的纯业务变换，Provider 只校验原输入身份、冻结新尝试的输入并负责执行。已有图片 URL、业务 hash 和当前提示词 builder 不能重建历史请求的原 inline 图片和精确 prompt；迁移使用下述原输入与派生接口，不能由 completion 命令替代。消费者接齐及验证前，旧实现继续保留。

#### 工作台读取原输入与提交业务派生

这两个入口仅向认证的内部管理员开放，并要求 `X-Workbench-Consumer: listing` 产生可信 `internal:listing` 来源；公司及外部入口不可使用。返回均禁止缓存。

- `GET /api/conversation-bindings/text-requests/{original_id}/retained-input`：纯读持久化输入，返回 `{protocol:"task-input-v1", kind:"text", request_id, input_hash, input}`。Provider 校验原 owner、请求 ID、输入 hash、绑定、账号、会话和提交时游标；不查询上游、不恢复或发送。不可导出的输入与身份不一致均拒绝，不能改用当前提示词重建。`input_hash` 是 Provider 的原输入身份，跨语言调用方应原样回传，不以 JSON 重新序列化替代它。
- `POST /api/conversation-bindings/text-requests/{original_id}/derived-successor`：普通模式接受下方七个字段。工作台计算业务内容；Provider 从原输入继承账号、绑定、会话、游标、图片和 `product_work` 的原 step。图片内容、顺序及 `image_ref` 标签不可改。`derivation_kind` 仅为业务审计名称，Provider 不按该名称运行商品规则。下方 `advanced_new_session` 是显式、独立的新会话后继模式，另需成功上下文引用。

```json
{
  "client_request_id": "stable-child-request-id",
  "original_input_hash": "<retained-input 返回的 64 位 hash>",
  "messages": [{"role": "user", "content": "工作台派生后的内容，保留原消息及媒体结构"}],
  "model": "auto",
  "thinking_effort": "high",
  "mode": "no_final",
  "derivation_kind": "workbench-business-rule-v1"
}
```

原会话接续模式为 `413_request_missing`、`no_final`、`incomplete`，仍须满足各自原有资格。普通 UNKNOWN、超时和暂停不能获得重发权限；内容审查后及实际发送前继续检查原结果、租约、会话顺序与原/子输入 hash。相同 child ID 和输入返回既有回执，不重复审查或发送；改内容必须拒绝。同一原请求仍只允许一个合法 child。

POST 返回既有文本回执，并附 `workbench_derived_input`：`{protocol:"workbench-derived-input-v1", original_input_hash, child_input_hash, mode, derivation_kind}`。内部管理员的后续 `task-result-v1` 纯读保留该审计字段；它与旧 `derived_input` 分开。工作台保存自己的业务派生证明，并按原业务身份幂等采用结果。新的 no-final/incomplete child 仍须通过原 ProductWork 关联检查才可关闭与归档，不凭生成成功直接关闭业务。

#### 原会话已进入其他讨论后的显式新会话后继

`advanced_new_session` 只接受已发送、没有原最终答案、本地执行结束且无活动恢复租约的
`REQUEST_CONVERSATION_ADVANCED`。原会话须有完整、无分叉、已完成的后续讨论证据；
创建时及实际发送前精确重读原图，新增消息、原答案迟到、身份或头节点变化均阻止本次发送。
它不将原请求改成成功或 `INCOMPLETE`，不提供未知请求的通用重发权限。

仍使用上述 endpoint 和七字段，设置 `mode:"advanced_new_session"`，额外提供：

```json
{
  "context_ref": {
    "request_id": "saved-gallery-request-id",
    "input_hash": "<gallery retained-input 返回的 SHA256>",
    "result_sha256": "<gallery 完整 content 原文 UTF8 的 SHA256>"
  }
}
```

成功上下文须属于同 owner、同账号、同原会话和同 ProductWork，其最终节点必须是原请求的提交父节点。
工作台把已存原资料、原图、完整成功上下文和本次业务指令组装到 `messages`；Provider 核对图片及
`image_ref` 的内容和顺序、成功结果原文、原输入 hash，不重新审核附图、不解释商品规则。
原 `original_input_hash` 仍指待接续的原请求。已成功首轮的冻结输入可以尚无物理会话字段；
输入内存在的身份字段仍须与成功回执一致，结果和原输入须能核实，才能导出或用作上下文。

工作台先持久保存一个稳定的 `client_request_id`。Provider 原子登记唯一后继，沿用原商品、步骤、
公平调度与账号，生成新的 client conversation，执行时创建新的 binding 和物理会话。
同 ID 同输入幂等返回；不同 ID 不能竞争出第二个后继，失败也不会自动产生第三次尝试。

回执另含 `advanced_session`，固定 `protocol:"advanced-session-successor-v1"`，及
`original_request_id/original_input_hash/original_account_identity/original_binding_id/`
`original_conversation_id/original_request_message_id/original_head_message_id`、
`completed_successor_evidence`、`context_ref`、`product_work`、`scheduling`、
`child_request_id/child_client_conversation_id`。
这些字段首次受理后冻结。`completed_successor_evidence` 含原 conversation/request、首个后续
request/提交父节点/final、`observed_at` 秒时间戳，可选 `tail_steps` 按序列出其余 request/final；
原 head 等于最后一个 final。`product_work` 复用 `product_ref/step_ref/fairness`；
`scheduling` 原样保留原输入值；当前 ProductWork 不接受非空 legacy workflow scheduling，
合法值为 null 或空对象，不能清空真实约束来绕过原门禁。原 request message/head 来自 Provider 回执及图证据，
不要求它们存在于 `retained-input.input`。

同一 proof 内的 `provider_binding_id/conversation_id` 两个字段在 queued 时可为 null，执行后
投影新回执身份；成功时必须非空、不同于原会话，且与回执相等。新账号必须仍等于原账号。
工作台核验上述关系后采用新 `parent_message_id` 游标，不再要求它等于旧附图节点。
内部纯读 `/api/task-results/text/{child_id}` 在 `receipt.advanced_session` 保留同一证明。

旧 UNKNOWN、旧输入、成功附图及后续讨论均保留。新旧尝试不重复计算商品名额，实际在途仍受原保护。
新后继发送后，旧答案迟到只保存于原请求，不覆盖后继；工作台只采用它记录的后继 ID。
完成关闭仍核验新后继及正常后续步骤，自动归档只处理新业务会话，旧讨论会话保留原归属和直接归档保护。
必须先部署两端兼容实现，再通过正式工作台入口验收原商品；代码和模拟测试不代表真实生成成功。

迁移验收复用 `test/test_category_directory_successor.py`、`test/test_required_attributes_successor.py`、`test/test_combined_copy_derivation.py`、`test/test_incomplete_compact_successor.py` 及现有 WBOZ combined-copy fixtures。尤其当前 structured copy 已是正式输入契约，应原样保留；迁移不同时压缩提示词、改变字段或替换图片。旧 receipt 与已创建 child 继续可读，不要求重新生成来升级协议。

## API

所有 AI 接口都需要请求头：

```http
Authorization: Bearer <auth-key>
```

<details>
<summary><code>GET /v1/models</code></summary>
<br>

返回当前暴露的图片模型列表。

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer <auth-key>"
```

<details>
<summary>说明</summary>
<br>

| 字段   | 说明                                                                                                         |
|:-----|:-----------------------------------------------------------------------------------------------------------|
| 返回模型 | `gpt-image-2`、`codex-gpt-image-2`、`auto`、`gpt-5`、`gpt-5-1`、`gpt-5-2`、`gpt-5-3`、`gpt-5-3-mini`、`gpt-5-mini` |
| 接入场景 | 可接入 Cherry Studio、New API 等上游或客户端                                                                          |

<br>
</details>
</details>

<details>
<summary><code>POST /v1/images/generations</code></summary>
<br>

OpenAI 兼容图片生成接口，用于文生图。

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <auth-key>" \
  -d '{
    "model": "gpt-image-2",
    "prompt": "一只漂浮在太空里的猫",
    "n": 1,
    "response_format": "b64_json"
  }'
```

<details>
<summary>字段说明</summary>
<br>

| 字段                | 说明                                                 |
|:------------------|:---------------------------------------------------|
| `model`           | 图片模型，当前可用值以 `/v1/models` 返回结果为准，推荐使用 `gpt-image-2` |
| `prompt`          | 图片生成提示词                                            |
| `n`               | 生成数量，当前后端限制为 `1-4`                                 |
| `response_format` | 当前请求模型中包含该字段，默认值为 `b64_json`                       |

<br>
</details>
</details>

<details>
<summary><code>POST /v1/images/edits</code></summary>
<br>

OpenAI 兼容图片编辑接口，可上传图片文件，也可按官方 JSON 格式传入图片链接并生成编辑结果。

```bash
curl http://localhost:8000/v1/images/edits \
  -H "Authorization: Bearer <auth-key>" \
  -F "model=gpt-image-2" \
  -F "prompt=把这张图改成赛博朋克夜景风格" \
  -F "n=1" \
  -F "image=@./input.png"
```

也可以直接传图片 URL：

```bash
curl http://localhost:8000/v1/images/edits \
  -H "Authorization: Bearer <auth-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-2",
    "prompt": "把这张图改成赛博朋克夜景风格",
    "images": [
      {"image_url": "https://example.com/input.png"}
    ]
  }'
```

<details>
<summary>字段说明</summary>
<br>

| 字段          | 说明                                            |
|:------------|:----------------------------------------------|
| `model`     | 图片模型， `gpt-image-2`                           |
| `prompt`    | 图片编辑提示词                                       |
| `n`         | 生成数量，当前后端限制为 `1-4`                            |
| `image`     | 需要编辑的图片文件，使用 multipart/form-data 上传           |
| `images`    | JSON 图片引用数组，支持 `{"image_url": "https://..."}` |
| `image_url` | 表单模式下也可直接传图片链接，支持重复字段传多张图                     |

<br>
</details>
</details>

<details>
<summary><code>POST /v1/chat/completions</code></summary>
<br>

面向文本、网页搜索与图片场景的 Chat Completions 兼容接口，不是完整通用聊天代理。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <auth-key>" \
  -d '{
    "model": "gpt-image-2",
    "messages": [
      {
        "role": "user",
        "content": "生成一张雨夜东京街头的赛博朋克猫"
      }
    ],
    "n": 1
  }'
```

<details>
<summary>字段说明</summary>
<br>

| 字段                   | 说明                                                                           |
|:---------------------|:-----------------------------------------------------------------------------|
| `model`              | 文本、搜索或图片模型；搜索模型会触发网页搜索兼容逻辑                                                   |
| `messages`           | 消息数组，支持文本、搜索和图片请求内容                                                          |
| `n`                  | 图片生成数量，按当前实现解析为图片数量                                                          |
| `stream`             | 文本、搜索和图片场景均支持，仍在测试                                                           |
| `tools`              | 文本场景支持 `web_search` / `web_search_preview` / `web_search_preview_2025_03_11` |
| `web_search_options` | 传入时会触发网页搜索兼容逻辑                                                               |

<br>
</details>
</details>

<details>
<summary><code>POST /v1/responses</code></summary>
<br>

面向文本、网页搜索和图片生成工具调用的 Responses API 兼容接口，不是完整通用 Responses API 代理。

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <auth-key>" \
  -d '{
    "model": "gpt-5",
    "input": "生成一张未来感城市天际线图片",
    "tools": [
      {
        "type": "image_generation"
      }
    ]
  }'
```

<details>
<summary>字段说明</summary>
<br>

| 字段       | 说明                                                                                      |
|:---------|:----------------------------------------------------------------------------------------|
| `model`  | 响应中会回显该模型字段，搜索和图片生成会走对应兼容逻辑                                                             |
| `input`  | 输入内容；搜索使用最后一条用户文本，图片生成需能解析出提示词                                                          |
| `tools`  | 支持 `image_generation`、`web_search`、`web_search_preview`、`web_search_preview_2025_03_11` |
| `stream` | 已实现，但仍在测试                                                                               |

<br>
</details>
</details>

## 社区支持

学 AI , 上 L 站：[LinuxDO](https://linux.do)

## Contributors

感谢所有为本项目做出贡献的开发者：

<a href="https://github.com/basketikun/chatgpt2api/graphs/contributors">
  <img alt="Contributors" src="https://contrib.rocks/image?repo=basketikun/chatgpt2api" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=basketikun/chatgpt2api&type=date&legend=top-left)](https://www.star-history.com/?repos=basketikun%2Fchatgpt2api&type=date&legend=top-left)


### Workbench Codex CLI 接入（待部署验收）

原有 `/v1/responses` 仍为 ChatGPT 文本兼容接口。新增 Codex 编程接口使用独立的
`/codex/v1/models`、`/codex/v1/responses`、`/codex/v1/responses/compact`，沿现有账号池和普通调用密钥，保留原生工具调用、工具返回与流式事件。
连接中心可以导入自己合法取得的 Codex `auth.json` 中的授权字段，管理自己的账号、刷新/停用，并分别显示图片容量及 Codex 上游用量窗口。导入方式不代表该账号已经通过对应路线验证；未知或读取失败不等于额度为零。

模型名称以实际查询为准。新会话只选择可用资源，已绑定会话不能通过换号重发恢复未知请求；没有内部充值、预算或额度分配。自定义 HTTP Provider 不等于 Codex Desktop 或所有云端功能已验证。
Mac / Windows 的隔离 CLI 启动脚本与恢复方法见 [Codex 客户端说明](docs/codex-client.md)。当前源码、模拟工具调用测试与公网真实验收分别记录，计划地址在生产部署及独立客户端验收前不能视为已可用。
