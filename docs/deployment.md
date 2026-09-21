# 部署与升级指南

本文介绍 ChatGPT2API 的常见部署方式，以及后续升级项目时需要保留的数据和执行步骤。

## 可靠共用与动态容量候选（2026-09-20）

本节对应原 `AI-POOL-STAGED-20260917` 的可靠共用增量；需求检查日期为2026-09-20，采用 Connection 原长期页及当前接管卡当日“调度及设置方案”。代码基于 Provider `35a4def7d02dd760f412bc27aa0653e4403f3812`，附件仅提供纯准入选择器与日志分析参考。下述为待协调独审、集成与正式发布的工程行为，不能当作现役生产验收。

### 同一资源账与动态派发

`PoolAdmission` 每秒重读原账号存储、原任务回执、原节拍文件和现有设置。普通 Key、公司登录、内部 Chat/图片调用及原生 Codex HTTP/SSE 都从原文本或图片任务进入同一准入事务。保存成功的新账号及恢复账号自动参与符合其授权、模型及额度的下一次选择，原等待无需换 Key、重启或重提。重复上游账号按原节拍身份去重，不增加容量。

| 资源 | 实际计算与含义 |
| --- | --- |
| Chat 活跃轮次 | 每个可用 Chat 账号最多1轮，正在读取的响应流占用该轮；保存的会话不计为轮次 |
| 图片生命周期 | 每号上限为 `min(image_account_concurrency, 该号可用图片额度)`；显式 `image_gen.remaining` 更低时采用更低值。尚未终结的原图片及 UNKNOWN 占位，原模型响应流结束后可释放轮次而继续占图片位 |
| 原生 Codex | 每个具备新鲜有效模型/额度观察的账号最多1轮，同时受原 `codex_max_concurrency` 约束；原会话绑定和待定响应同样限制领取 |
| 节拍与冷却 | 账号请求间隔、模型轮次发送间隔、上游冷却分别取下一可发时间；有空位不等于现在可发送 |
| Provider 技术保护 | 原 Chat 文本执行器4个、同时执行输入256MiB与传输 body-reader 独立约束；磁盘等待不占执行线程，不等于账号配额 |

账号停用、异常、授权或模型额度不可用会停止相应新派发；提交前再次核对。已提交/UNKNOWN 原任务不改号、不释放为“可重发”，恢复只读原结果。模型特定限制只影响其模型；Codex额度不能代替 Chat图片额度。缺失占用用 `null`，已知没有占用用 `0`；未知额度使该路线不准入，但不会伪造一条额度为0的观察。

原 `/api/workbench/ai/pool/resources` 投影增加 `accounts[]`：每行 `provider_account_identity`、`enabled`，以及 `chat_turn` / `image` / `codex` 的 `capacity, occupied, free, dispatchable_now, next_at`。汇总保留 `slots_total/inflight/slots_free`，增加 `dispatchable_now`、`queue.mode=durable_original_receipts`、`queue.queued/by_source`、`execution.active_input_bytes/max_input_bytes/chat_workers_active/chat_workers_limit`。这是调度器同一存储的只读投影；没有改 UI 或账号身份展示作者负责的 `owned_accounts.py`。

基础公平以认证后的来源轮转，再在合格账号间轮转，同来源保留先到顺序和同会话前序。普通 Key 的来源取真实 Key ID，公司来源取已验证公司身份；私网管理员调用仅接受服务端允许的消费者名。Workbench 配套给两方向上新的共同 Worker 标记 `internal:listing`，给 Content 标记 `internal:content`；Happy 沿原公司身份，未修改其本机传输。普通客户端自报来源/优先级不改变调度，不新增预算或权限。

### 原任务持久化、领取与恢复

沿用 `data/text_tasks.sqlite3` 的原 `requests` 表，原图片 JSON 收据一次导入同库 `image_requests`，原 JSON 保留为迁移前历史。恢复所需实际文本/参考图 bytes、原请求身份和必要转发字段写入同目录 `text_tasks_inputs/`（目录0700，文件0600，受控文件名、无符号链接），先 fsync 输入，再提交原 owner/ID/输入一致性记录。公开回执不暴露私密输入、路径或账号凭据；重复同 ID 同输入只读原记录，不同输入409。

领取使用 SQLite `BEGIN IMMEDIATE`：同时核对所有原任务占用、账号状态、顺序，保存原账号绑定与30秒领取租约；执行中每10秒续租。账号 JSON 及完整模型响应流节拍使用进程间文件锁。所有进程必须共享同一本机数据根、同一版本及支持 POSIX 锁的文件系统；验证覆盖本机多进程，不宣称跨主机/NFS或其他账号存储后端已验证。

准入保留已有 `provider_account_identity`。尚无此图片/会话身份字段的合法账号（包括旧、新 Codex-only）使用原 `managed_pool_account_ref` 派生同一个稳定、不含凭据的身份，正常读取不回写账号；以后创建普通会话绑定时保存的也是同一身份。原 pool ref 随既有凭据轮换持久保留，多进程读取沿它接续令牌别名；原 Codex affinity、response 归属及 UNKNOWN 不改绑，不要求生产手工补字段或先做一次图片调用。

模型 POST 前持久记录“可能已发送”并再次检查原领取。租约过期而尚未开始发送的任务重新等待；可能已发送的任务转原 UNKNOWN，不能换号补发。公共持久 `/api/chat-requests`、内部持久 `/api/conversation-bindings/text` 及 `/api/image-tasks` 按原账号、会话及请求消息查询。持久图片已保存结果 ID 时仅重新下载，不调用生成。原生 Codex 已落盘的原始输出可按同请求 ID重新订阅，缺少权威原结果读取能力的 Codex UNKNOWN 保留待查，不发起替代请求。无稳定请求ID的旧客户端不能获得跨客户端重试去重保证，应采用更新后的调用合同。

既有 PPT/PSD 文件任务也经同一原文本任务存储准入，原文件任务查询接口投影同一收据；等待输入重启可恢复，旧文件 JSON 记录不重建。文件任务 UNKNOWN 没有新增自动导出/重生成路径，也不冒称支持图片任务的原结果下载恢复。搜索以其实际固定模型检查可用性。

旧版原图片收据若没有可恢复输入或原请求消息边界，保留旧记录/明确未知状态，不从 hash、prompt相似度或会话末尾猜测重建。兼容接口的多图调用顺序运行显式槽位，已发送槽位不能二次发送；公开持久图片任务仍为一任务一图。

直接兼容 `/v1/chat/completions`、`/v1/responses`、`/v1/messages` 的普通文字路径现在保留原 Chat 历史、实际发送根节点、原用户消息 ID、流中返回的会话 ID 和原协议响应 ID。中途 EOF 不再生成成功结束事件；重启后沿同一原回执的恢复租约做有界原账号 GET，唯一完成结果通过原协议格式化并落入新的私密完整响应文件，再原子更新该回执。已打开的订阅固定读取原前缀文件，不把新完整响应拼到旧前缀后；同 ID 重连取得完整结果，原响应/工具调用 ID 保留。恢复不调用生成处理器，不重复写一次调用日志。恢复次序复用原 retry 时间，避免短退避的旧扫描长期挤压其他到期读取。没有完整定位和格式元数据的旧兼容收据仍不进入这个文字恢复器。

`/v1/search` 及 Chat/Responses 的搜索工具路径也沿上述原回执恢复：准备和模型提交使用同一个已持久化用户消息 ID，流中取得的会话 ID 立即保存。没有收到会话 ID 时沿原账号做有界消息定位；只采用原请求唯一终结分支的答案和引用，不取会话最新回答、不以片段稳定或超时判成功。恢复保留原 Responses 搜索调用 ID、消息 ID 和响应 ID，复用同一私密完整输出发布方式。准入检查搜索实际发送的固定模型 `gpt-5-5`，实际模型只投影到原回执的调度字段，原输入封装和哈希保留客户端别名；升级后同ID同输入可继续读旧成功/UNKNOWN结果，不因内部模型映射收到409。启动时校验已保存输入后，只修正尚未领取的旧排队搜索的调度模型，并以原回执比较更新防止覆盖并发领取；该模型额度恢复后原等待请求可重新领取。实际输入变化仍冲突。此项有隔离测试，尚未做真实上游断线恢复验收。

未走公开持久图片适配的 Chat 路线 `/v1/images/generations`、`/v1/images/edits`、Chat 图片和 Responses 图片现在也使用原文本回执的恢复租约。执行前保存格式参数和每图原消息 ID，执行中保存原会话、结果 ID 及私密成果文件；SQLite 备份同时保留这些文件。恢复只读原账号的原分支，已有结果 ID 时直接解析原下载地址，任何原地址失败都不把部分图片报成功。原分支已终结而下载失败时可释放模型轮次，图片生命周期与原结果继续等待；仅有图片文件而无终结证据时继续保留轮次。下载成功成果不重画，后续恢复复用本地文件。

多图中途失败后，先由同一恢复租约确认并保存已发送图，再将原回执的未发送余项放回原准入调度；账号恢复后原请求自动续行，仍绑定原账号并参与来源轮转。过期恢复 worker 不能覆盖新领取结果。完整协议响应从保存的成果重新整理，保留已发出的响应 ID；旧订阅只读原前缀，同 ID 重连取得新的完整响应。恢复记录区分读取原会话、解析下载地址、下载、保存和格式化阶段；ChatGPT GET 的429保留原请求关联及账号冷却证据，对象存储下载429仅退避下载、不使ChatGPT账号冷却。以上为隔离测试证据，尚未做真实上游图片断线/重启验收。

尚未完成的 P3 范围：原生 Codex UNKNOWN 的完整上游结果读取。原生 Codex 在 `response.created` 时保存真实上游 response ID 和可得 HTTP 请求 ID，但没有凭此虚构可用的 GET 恢复端点。原生 Codex UNKNOWN、缺少定位/格式元数据的旧 raw 图片收据仍只重读已落盘响应或前缀，不补造尾部、不重发；不能称所有兼容路径的原结果恢复已完成。带公开图片标记的同步图片接口继续沿 `/api/image-tasks` 原收据查询和恢复下载。

结果 UNKNOWN 与模型轮次是否结束分开计算：原 GET 只有在原用户的唯一完整分支上证明最终助手消息 `finished_successfully/end_turn=true`、且没有活动或歧义分支时，才可保存轮次结束证据；空结果仍保持 UNKNOWN 及同会话顺序，其他独立等待可使用已经证实释放的轮次。旧公开 Chat 收据只有明确 `stream_open/http/conversation/422` 的原模型拒绝证据可直接排除模型占用，状态和原输入身份不改写。`failed/RESULT_UNRECOVERABLE` 若仍为 `upstream_outcome=unknown`，同样保留轮次和同会话保护，并继续有界查询原结果；结果读取失败或读取次数不能替代原执行终结证据。超时、缺失结果、阶段不明的422、任务年龄均不能释放；图片生命周期占用仍另算。

本次真实只读范围（2026-09-21）：现役仍为 Provider42，候选基线 `a8697d8036edf9e8c34815147a772cfa785b40d1`。原 `inv-g8wggugvrj` 对9个指定 UNKNOWN 最多12次 GET、0 POST，9条原回执字节不变，仍不能归属原结果；账号文件同期有变化，不能将其宣称为未变或归因于本读取。后续 P6 新原请求使现场成为512文本/10 UNKNOWN；`inv-b8wh180jme` 仅查询这10个原ID的白名单诊断，0 Provider/GET/POST/写。其中2条有完整模型入口422证据，2条422缺阶段、2条timeout/stream_open、4条缺诊断。因此此代码对这组证据只能排除2条模型占用，其余8条继续未知，不能宣称已解除生产迁移容量阻塞。这10条是status=unknown样本；精确副本预检还须把旧failed/RESULT_UNRECOVERABLE且upstream_outcome=unknown的记录计入，不能将这10条当作全部未知执行。原精确制品预检 `inv-d8wfs006gr` 的236图/217成功/509文本保存证据与Chat可派发0的阻塞保留；本轮没有生产切换、原任务重发或新图生成。

协议在发送前明确拒绝（如空消息HTTP400）保存 `failed/PROTOCOL_REQUEST_REJECTED` 和原HTTP错误语义，不变成UNKNOWN或无限重排。只有明确的可恢复容量/领取暂不可用才等待。兼容执行者通过原 `LoggedCall` 在结果落盘或失败/流结束时记录一次调用，保留Key、原请求ID、实际账号及可得429证据；HTTP订阅断开不取消记录，同ID重连不重复写成功日志。

部署时须由协调者按 Workbench 当前 runbook 和不可变 digest，先完成候选制品预检与可恢复备份，停止旧 Provider 写入者后再导入/启动，不能让仍写 `image_tasks.json` 的旧版本和新版本混跑。现有备份的 `image_tasks` 选项包含一致的 SQLite 副本和所有引用的私密输入、已提交输出前缀；它们应与原账号、节拍、图片存储整体保护。回退保留切换后新增/修改的 SQLite 记录与输入，不得直接用旧 JSON 覆盖回退或删 UNKNOWN 后重画；旧版本不理解新等待记录，故必须先停止新写入者，逐项核对待处理记录和恢复路径。此次没有执行迁移、部署或生产参数调整。

### 429 分层

| 来源 | 证据与处理 |
| --- | --- |
| 公司入口 | 配套 Content API 返回 `company_transport` / `before_provider_acceptance`、传输请求ID和 Retry-After；该次完整受理尚未发生，按原 ID核对，不使账号 cooldown |
| Provider 本地容量 | body-reader 返回 `provider_capacity` / `before_acceptance`，原 Codex 本地保护标为 `provider_capacity` / `admission`；已完整受理的池满请求持久等待，不因池满丢失或更换上游账号 |
| ChatGPT 上游 | `upstream_chatgpt`，区分 `http_429` 与 HTTP200中的 `sse_rate_limit`，保存原任务关联、阶段、可得上游请求ID、Retry-After秒数和冷却时间。只作用于相应原账号节拍 |
| Codex 上游 | `upstream_codex`，区分 Responses HTTP/SSE，保存相同证据与原模型额度状态；沿原观察刷新周期与 Retry-After 退避，不借其他账号重发 UNKNOWN |

这些证据在原回执及安全日志中保留，不能从客户端见到“429”直接推导限额来源。HTTP200中的普通助手文字不被当作 SSE限流；上游未提供请求ID或Retry-After时保留缺失/回退冷却事实，不能伪造上游值。

### 4账号的分时观察与推荐运行范围

只读现场 `inv-b8wcdsg967`，时间2026-09-20 22:53:41Z：当时现役 `35a4def7`、digest `sha256:f380bb3160771f2d0a1c5c4c46ad40dc3c22fc2401f5a89a149f188b74f2ffb3`，早于另批PR42发布。4条原账号中2条为 Codex-only，2条为 Chat OAuth Pro；后两条22:50观察图片额度各999。配置为账号请求10秒、模型消息60秒、每号图片4、Codex服务器4、刷新5分钟，均未更改。

协调者追加只读 `inv-d8wepk0d21`（2026-09-21 00:12:27–33Z，现役Provider42）：两条原Codex-only账号没有 `provider_account_identity`，均有原稳定pool ref；其中第2条主额度allowed=true、used51%、pending/unknown为0。候选f1d585a曾因此跳过合法账号；本次沿原pool ref修正，隔离JSON账号、真实准入器和原生Codex执行器验证原账号可用、新账号自动接走原等待、跨worker轮换及重启原绑定。该只读证据不是新的模型发送样本，也不表示本候选已部署或生产动态容量已验证。

协调者原 `inv-d8waiv0pxh` 的21:50:50Z快照：Codex-only两条的Codex周窗分别用满100%、使用34%；两条Chat Pro的Codex观察也为周窗100%（其观察时间20:48）。这是旧观察，超过5分钟后不能当作当前可派发量；独立附加模型窗口0%不抵消Codex主周窗用满。该版本没有权威的每号持久占用字段，不能把缺失填0。

| 账号/模型/操作 | 输入规模与样本 | 推荐保留的运行范围 | 429来源证据 |
| --- | --- | --- | --- |
| 两条 Chat Pro / `gpt-image-2` / 生图 | 原历史收据235条（216成功、19错误），跨历史版本；当前导出仅1条发送节拍事件，缺模型/输入/当时并发，不能把两组拼成吞吐或成功率 | 每号0–1活跃模型轮次；未终结图片0–4且受真实额度约束。两号理论至多8图片生命周期位，不是8轮同时发送；HTTP间隔至少10秒，模型轮次至少60秒并服从更长cooldown | 当前容器有界1255行导出没有匹配限流事件；不证明历史或未来无429 |
| Chat文本/识别 / 实际目录模型 | 本次没有可按模型、输入规模、并发归属的成组样本 | 同号共享上述模型轮次与节拍，不能给文字和图片各算一套容量 | 当前导出不足以分别量化 |
| Codex / 明确选择的可用模型 / Responses | 本次只有授权/额度只读观察，没有新模型执行样本 | 每号0–1轮，服务总量受现有4及新鲜可用账号数共同限制；不建议提高原参数或推导固定安全发送间隔 | 三条历史额度受限不等于三次现场HTTP429；真实HTTP/SSE限流须查其原响应证据 |

唯一节拍样本的间距为10048.7秒、记录的最低间隔60秒，不能据它断言60秒已充分采样。以上是保留现有配置的保守运行边界，**按账号/模型/操作/输入规模测得的安全并发及发送间隔范围尚未建立**；没有证据支持扩大。正常业务发布后应从新增安全关联日志收集样本，再对一个参数做已授权的最小验证；不要故意撞429或批量压账号。4→5账号、16→20图片生命周期位仅为独立数据/假时钟回归，不是生产配置授权，也不套用正式OpenAI API RPM/TPM。

工程覆盖：持久满载等待、新账号/停用/恢复、可信来源公平、原会话与账号绑定、输入重启恢复、原子多进程竞争、过期领取防重发、UNKNOWN不重发、原结果下载恢复、三类认证入口共享占用及原生输出断线续读。真实多消费者共用、新账号线上自动唤醒、生产重启与推荐区间实测均待正式发布后的原授权验证；不以离线通过结案。

## 工作台程序密钥政策第二阶段

**2026-09-17：第二阶段已上线并获用户确认。** Provider PR18与Workbench PR948已发布；按用户明确决定，两把保留旧密钥允许Chat/Codex，另外三把旧密钥已撤销。原政策迁移已完成，后续发布不得重复迁移、恢复旧auth快照或启动policy-unaware版本；新增key按其实际政策执行。下文保留维护与恢复边界。

政策写入既有 auth_keys 记录，不新建账号池或密钥正本。政策为 `version:2`，`routes` 可选 `chat`、`codex` 或两端；不再有生图/识别/规划/编程子权限。版本1仅为未发布候选，不自动扩大转换；接口/工具技术就绪与密钥授权分开。内部 Workbench 管理接口保留管理员认证和服务端 owner 注入，普通程序密钥不能调用：

- `POST /api/workbench/ai/keys`：`name`、`routes`，原始密钥只返回一次。
- `PATCH /api/workbench/ai/keys/{id}/policy`：`routes`、`expected_revision`，仅当前 owner；旧无政策记录的初始版本为 0。冲突返回 409，非法端列表返回 422。
- `GET /api/workbench/ai/keys`：脱敏端级政策；不返回 hash、owner 或原始密钥。
- 既有撤销接口保持不变。每次认证读取当前存储；最后使用时间更新不能覆盖并发撤销或缩小用途。

JSON 使用同路径文件锁与原子替换；SQL 使用事务（SQLite 写锁、PostgreSQL advisory lock、MySQL 同连接 named lock）；Git 使用本地文件锁和非强推，远端竞争失败即报错。已有旧进程不会遵循新事务规则，因此不得滚动混跑新旧 auth_keys 写入者。

旧密钥核对文件只使用ID与允许端，不含原始密钥，不按名称猜测，也不再配置端内模型/操作例外。例如：

```json
[{"id":"<confirmed-existing-key-id>","routes":["chat"]}]
```

`python scripts/reconcile_program_keys.py --assignments <reviewed-file>` 默认只读核对；必须一次覆盖全部启用且无政策的普通密钥，集合变化即失败。`--apply` 才写入。不再使用历史文本用途例外；允许Chat即可使用该端已支持操作。缩小允许端仍允许读取原有且同 owner 的任务/文件/回执，不重新生成，撤销则禁止全部调用。旧直接cursor的归档也仅限admin/Content；普通客户端直接绑定会话提交尚不支持，返回技术501，不能凭别人的binding写入。旧 `GET /api/conversation-bindings/text` 的直接 cursor 没有持久调用方归属，仅保留现有 admin/Content 路径；原内部回执继续保留。第三阶段新增公共 `/api/chat-requests` 提交与 `/{request_id}` 查询、`/{request_id}/recover` 原结果读取，使用同一TextTaskService按key ID归属；客户端不提供账号或cursor，不能凭他人的cursor取得文本。只精确开放这些公网路由及无秘密 `/ai/integration.md`，不得公开管理路由或整个 `/ai/`。

公共 Chat 的 `GET /api/chat-requests/{id}` 和 `POST /api/chat-requests/{id}/recover` 可能返回 `status=failed`、`error_code=CHAT_RESPONSE_NOT_TEXT`：仅在原 user 的唯一已完成分支中确认图片工具产物、最终文字为空且无活动分支时成立。`result={"type":"non_text","artifact_type":"image","artifact_count":1}` 表示上游产物引用数量，不代表已经下载、图片质量通过或 Happy 图片任务成功。`recovery.reason=REQUEST_RESULT_NON_TEXT`、`upstream_outcome=completed`、`retryable=false`、`requires_new_conversation=false`；客户端停止自动恢复/重生成，保留原请求交由原消费者处理协议不符。同 ID 同输入仍只读原结果，不同输入冲突；换新 ID 也不是被授权的恢复方式。普通空文本、活动分支和无法归属继续沿现有 UNKNOWN 查询规则处理。

原图片引用与 conversation/request/tool/final 的对应关系保存在同一 `text_tasks.sqlite3`、同一 owner/request 回执的内部 `_non_text_result` 字段；普通与管理投影都不输出原资产地址。没有新增下载接口或复制任务正本。发布和回退继续保留整个原数据库，不能删掉这些原产物引用来重试。

正式切换遵循 Workbench 当前 `docs/runbooks/DEPLOYMENT_AND_ROLLBACK_P0_1.md`、唯一发布负责人及精确镜像 digest：先保存密钥存储与现役版本回退点，停止旧 Provider 写入者，使用已核对的完整映射迁移原记录，启动新 Provider 后发布兼容 BFF/页面，再独立读回。不得修改账号/任务正本或重建未变服务；生产预检还需验证镜像 Python 3.13 运行态。回退必须先停止新写入者，核对切换期间新增/修改/撤销的密钥，禁止盲目覆盖旧快照导致已撤销密钥复活。任何未知结果先读回，不重复迁移。

## 部署前准备

服务器需要安装：

- Docker
- Docker Compose v2
- Git

首次部署前建议确认：

```bash
docker version
docker compose version
git --version
```

项目核心持久化文件：

| 路径 | 作用 |
| --- | --- |
| `config.json` | 主配置、后台密钥、代理、图片、备份等配置 |
| `.env` | Docker compose 环境变量 |
| `data/` | 账号、日志、图片、任务记录等运行数据 |

升级和迁移时重点保留以上内容。

## 方式一：普通 Docker 部署

适合不需要 WARP / FlareSolverr 清障的场景。

```bash
git clone git@github.com:basketikun/chatgpt2api.git
cd chatgpt2api
```

设置 `config.json` 中的 `auth-key`，或在 `docker-compose.yml` 中配置：

```yaml
environment:
  - CHATGPT2API_AUTH_KEY=your_secret_key
```

启动：

```bash
docker compose up -d
```

访问：

```text
http://localhost:3000
```

API 基础地址：

```text
http://localhost:3000/v1
```

查看日志：

```bash
docker logs -f chatgpt2api
```

停止：

```bash
docker compose down
```

## 方式二：WARP / FlareSolverr 部署

适合上游请求经常遇到 Cloudflare 拦截的场景。该方式会启动：

- `warp-proxy`
- `privoxy`
- `flaresolverr`
- `init-config`
- `app`

复制环境变量模板：

```bash
cp .env.example .env
```

至少修改 `.env` 中的：

```text
CHATGPT2API_AUTH_KEY=your_secret_key_here
```

启动：

```bash
docker compose -f docker-compose.warp.yml up -d --build
```

访问：

```text
http://localhost:3000
```

FlareSolverr 相关配置可以在后台设置页的 `FlareSolverr` tab 中查看和测试。

查看容器状态：

```bash
docker compose -f docker-compose.warp.yml ps
```

查看日志：

```bash
docker logs -f chatgpt2api-warp
docker logs -f chatgpt2api-flaresolverr
```

停止：

```bash
docker compose -f docker-compose.warp.yml down
```

## 方式三：源码运行

适合本地开发或临时调试。

后端：

```bash
git clone git@github.com:basketikun/chatgpt2api.git
cd chatgpt2api
uv sync
uv run main.py
```

前端开发服务：

```bash
cd web
bun install
bun run dev
```

源码方式运行时，后端默认读取项目根目录的 `config.json` 和 `data/`。

## 存储后端

默认使用本地 JSON 文件：

```text
STORAGE_BACKEND=json
```

可选值：

| 值 | 说明 |
| --- | --- |
| `json` | 本地 JSON 文件，默认方式 |
| `sqlite` | 本地 SQLite，通常存放在 `data/accounts.db` |
| `postgres` | 外部 PostgreSQL |
| `git` | Git 私有仓库存储账号数据 |

PostgreSQL 示例：

```yaml
environment:
  - STORAGE_BACKEND=postgres
  - DATABASE_URL=postgresql://user:password@host:5432/dbname
```

SQLite 示例：

```yaml
environment:
  - STORAGE_BACKEND=sqlite
  - DATABASE_URL=sqlite:////app/data/accounts.db
```

## 升级前备份

升级前建议备份：

```bash
mkdir -p backups
tar -czf backups/chatgpt2api-$(date +%Y%m%d-%H%M%S).tgz config.json .env data
```

如果没有 `.env`，可以去掉：

```bash
tar -czf backups/chatgpt2api-$(date +%Y%m%d-%H%M%S).tgz config.json data
```

也可以在后台设置页配置 Cloudflare R2 备份，用于定时备份关键数据。

## 升级：普通 Docker 部署

进入项目目录：

```bash
cd chatgpt2api
```

备份：

```bash
mkdir -p backups
tar -czf backups/chatgpt2api-$(date +%Y%m%d-%H%M%S).tgz config.json .env data
```

拉取最新代码和镜像：

```bash
git pull
docker compose pull
docker compose up -d
```

查看状态：

```bash
docker compose ps
docker logs -f chatgpt2api
```

## 升级：WARP / FlareSolverr 部署

进入项目目录：

```bash
cd chatgpt2api
```

备份：

```bash
mkdir -p backups
tar -czf backups/chatgpt2api-$(date +%Y%m%d-%H%M%S).tgz config.json .env data
```

拉取最新代码并重新构建：

```bash
git pull
docker compose -f docker-compose.warp.yml up -d --build
```

查看状态：

```bash
docker compose -f docker-compose.warp.yml ps
docker logs -f chatgpt2api-warp
```

## 升级：源码运行

```bash
cd chatgpt2api
git pull
uv sync
```

如果需要重新构建前端静态产物：

```bash
cd web
bun install
bun run build
```

然后按你的进程管理方式重启后端服务。

## 回滚

如果升级后需要回滚代码：

```bash
git log --oneline -n 20
git checkout <旧版本commit>
```

普通 Docker 部署：

```bash
docker compose up -d
```

WARP / FlareSolverr 部署：

```bash
docker compose -f docker-compose.warp.yml up -d --build
```

如果需要恢复数据：

```bash
tar -xzf backups/你的备份文件.tgz
```

恢复数据前建议先停止容器，避免运行中写入覆盖：

```bash
docker compose down
```

或：

```bash
docker compose -f docker-compose.warp.yml down
```

## 常用维护命令

查看容器：

```bash
docker compose ps
```

查看主服务日志：

```bash
docker logs -f chatgpt2api
```

查看 WARP 部署主服务日志：

```bash
docker logs -f chatgpt2api-warp
```

重启普通部署：

```bash
docker compose restart
```

重启 WARP 部署：

```bash
docker compose -f docker-compose.warp.yml restart
```

清理未使用镜像：

```bash
docker image prune
```

不要直接删除 `data/`、`config.json`、`.env`，除非已经确认有可用备份。
