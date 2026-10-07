# 部署与升级指南

本文介绍 ChatGPT2API 的常见部署方式，以及后续升级项目时需要保留的数据和执行步骤。

## 启动时的模型目录就绪状态

`/health` 通过只证明账号与存储的基础健康，不保证上游模型目录已探测完成。
`GET /v1/models` 的未知或部分目录沿用 `model_catalog: {"state":"partial"}`，与该次目录快照一起返回；已观测模型保持可见，图片别名存在不代表文字目录已经完整。若 partial 私有目录为空，或公开目录没有任何具有观测证据的模型，返回 `503`、`detail.code="MODEL_CATALOG_PENDING"` 和 `Retry-After: 1`。公开图片别名仍需既有 `image_gen` 观测证据；已知的空目录仍返回正常 `200`，不伪报临时等待。

启动客户端仅对上述明确临时状态，或“目标模型缺失且目录为 partial”，在自身期限内有限重读目录。partial 不能永久缓存；已知完整目录缺少配置模型仍应报告配置错误。该规则只适用于目录 GET，不允许重试模型生成 POST、替换模型、账号或原会话。

## 商品图片对话接续（image-thread-v1）

该合同是原持久图片任务的可选接续能力，不新增队列、账号池或数据库。控制文字会话与图片会话分层；同一商品的新图片请求通过同一 `image_thread_id` 串行承接原图片会话，同一身份下不同商品仍可独立调度。它不是把每张图的聊天在UI里隐藏，也不是所有文字与图片已经合成一条上游对话。

发布顺序：先发布支持本合同的Provider并核对实际版本，再安装匹配Happy连接器。原 `GET /v1/models` 在原调度器可用时返回 `service_capabilities.image_thread="image-thread-v1"`；客户端在接受新的线程任务前读取该能力，不支持时应停止，不能静默回退新聊天。新能力只用于原 `/api/image-tasks/generations` 与 `/api/image-tasks/edits`，同步兼容图片接口拒绝线程字段。

- `client_task_id`：每次生成或修改独立且不可变；断线后查询这个原编号，不能换编号重发。
- `image_thread_id`：调用者自己选择的稳定工作编号，1至200个安全字符，在已认证的原owner范围内隔离。不需要登记应用名单、分配预算或传上游账号/对话号。
- `edit_source_task_id`：修改时指定原成功图片任务；`edit_source_index`是本轮参考图中该原图的位置（从0起）。服务器必须核对同owner、同线程/明确旧根及真实图片字节。改主图时即使其后已有卖点图，仍以指定主图为底图，但从线程最新已完成位置接续，不从主图旧parent岔出另一分支。

每次受理在原SQLite事务中取得唯一前序。原回执公开 `image_thread`（protocol、id、previous_task_id、edit_source_task_id），不公开私有账号和上游游标。忙或前序未结束时保留原输入排队，不提前占执行位置。派发及发送前再次核对前序与修改源；上游读取还需证明原请求的唯一完成分支及最新parent。未知、停止、缺记录、源版本变化或外部新消息均不允许换号、新开聊天或猜最新回答。等待原因保存在原waiting投影中。

图片文件已经生成但最后一轮尚未证明结束时，保留原结果定位，只读恢复；完成结果与后继准入必须分别核对。新线程任务的恢复仍走原resume-poll/下载路径，成功后的原请求再次提交只读原结果。同一图源回执若后来改变，已排队的修改不能悄悄跟随新版本。

已有图片任务不迁移、不改hash、不重画。确有完整身份、原请求及已保存图片的旧成功任务，可以在新修改任务中作为明确旧根，实际续接它原来的上游会话；旧根本身保持原记录。多个历史散落图片会话不会被追溯合并。没有这些证据时明确拒绝接续，不退回另开聊天。

工程测试包括真实原SQLite、原准入/绑定生成代码及受控上游，不能冒充真实ChatGPT、Mac安装、图片质量或完整自动/手动生产验收。55条被明确停止的恢复记录保持停止。生产切换和业务调用仍按Workbench既有授权及精确发布runbook执行，本节不是生产操作授权。

## 可靠共用与动态容量候选（2026-09-20）

本节对应原 `AI-POOL-STAGED-20260917` 的可靠共用增量；需求检查日期为2026-09-20，采用 Connection 原长期页及当前接管卡当日“调度及设置方案”。代码基于 Provider `35a4def7d02dd760f412bc27aa0653e4403f3812`，附件仅提供纯准入选择器与日志分析参考。下述为待协调独审、集成与正式发布的工程行为，不能当作现役生产验收。

### 同一资源账与动态派发

`PoolAdmission` 每秒重读原账号存储、原任务回执、原节拍文件和现有设置。普通 Key、公司登录、内部 Chat/图片调用及原生 Codex HTTP/SSE 都从原文本或图片任务进入同一准入事务。保存成功的新账号及恢复账号自动参与符合其授权、模型及额度的下一次选择，原等待无需换 Key、重启或重提。重复上游账号按原节拍身份去重，不增加容量。

| 资源 | 实际计算与含义 |
| --- | --- |
| Chat 活跃轮次 | 每账号按 `chat_account_concurrency`（默认仍1）计活动轮次；正在读取的响应流占位，保存的会话不计位。配置入口已支持先验证2路，不在代码升级时自动提额 |
| 图片生成 | 每号上限为 `min(image_account_concurrency, 该号可用图片额度)`；显式 `image_gen.remaining` 更低时采用更低值。UNKNOWN保留占位；原任务已持久化结果ID和生成结束证明后，下载/保存不继续占生成位。多次模型发送的原多图请求在整个已知序列结束前保留占位 |
| 原生 Codex | 每个具备新鲜有效模型/额度观察的账号最多1轮，同时受原 `codex_max_concurrency` 约束；原会话绑定和待定响应同样限制领取 |
| 节拍与冷却 | 账号请求间隔、模型轮次发送间隔、上游冷却分别取下一可发时间；有空位不等于现在可发送 |
| Provider 技术保护 | 共用调度的文字执行数随可用物理账号轮次计算，移除原全局4限制；图片执行器允许生成和保存重叠，但总在途工作不超过可用图片容量的2倍，防止慢下载无限累积线程。同时执行输入256MiB和原传输 body-reader 限制保留；磁盘排队不占执行线程 |

账号停用、异常、授权或模型额度不可用会停止相应新派发；提交前再次核对。已提交/UNKNOWN 原任务不改号、不释放为“可重发”，恢复只读原结果。模型特定限制只影响其模型；Codex额度不能代替 Chat图片额度。缺失占用用 `null`，已知没有占用用 `0`；未知额度使该路线不准入，但不会伪造一条额度为0的观察。

原 `/api/workbench/ai/pool/resources` 投影增加 `accounts[]`：每行 `provider_account_identity`、`enabled`，以及 `chat_turn` / `image` / `codex` 的 `capacity, occupied, free, dispatchable_now, next_at`。汇总保留 `slots_total/inflight/slots_free`，增加 `dispatchable_now`、`queue.mode=durable_original_receipts`、`queue.queued/by_source`、`execution.active_input_bytes/max_input_bytes/chat_workers_active/chat_workers_limit`。2026-09-27候选同时投影原账号安全引用、可派发状态、队列等待原因/最长等待、核查原结果/图片保存数和图片执行器占用。Workbench配套显示真正可启动数、异常账号和前驱阻塞；旧服务缺字段显示未知，不能据余量推定可派发。未另建账号或任务数据库。

基础公平以认证后的来源轮转，再按来源内owner轮转并在合格账号间轮转；同owner按接受顺序，同会话前序保留。游标保存在原SQLite runtime行，重启不归零。普通Key的来源取真实Key ID，公司来源按已验证公司+用户归组，多连接器保留各自原回执owner而不会增加这个用户的派发份额；私网管理员调用仅接受服务端允许的消费者名。Workbench 配套给两方向上新的共同 Worker 标记 `internal:listing`，给 Content 标记 `internal:content`；Happy 沿原公司身份，未修改其本机传输。普通客户端自报来源/优先级不改变调度，不新增预算或权限。

### 2026-09-27接续：占位、公平与真实验收

对应用户对五项补齐计划的“补吧”。本候选基于已含认证异常恢复PR64的main b0e878e，保留原Provider PR63终结空响应后继方案的独立范围；未修改线上并发值、10/60秒发送节拍、116 HOLD或55条停止恢复标记。Workbench配套取消整款四位预算，只保留同商品互斥和旧回执恢复；各入口仍调用原Provider持久调度。

前驱结果区分缺失、正在执行、失败、UNKNOWN、图片保存待完成、未确认和人工停止恢复；公开等待信息包含同owner的 `previous_task_id` 与 `read_original_predecessor` 操作提示。这里只查原记录，不因明确失败或UNKNOWN自动新建会话、换账号、重发生成。新任务只在前驱确已保存且原绑定确认后继续。

真实验收仍是 `waiting-external`，依赖唯一发布协调部署准确Provider/Node/Web工件。离线多进程、重启、断线、账号停用恢复、满队列、公平、2/4路容量和UI证据只证明工程行为，不能替代以下步骤：

1. **准确上线与原负载：** 读取运行digest、资源设置revision和健康账号；选择原已授权、未HOLD且没有UNKNOWN重发风险的商品队列。记录原owner/request/account/conversation身份、初始排队和已保存结果；不为压测新增无业务用途图片。
2. **每账号两路：** 通过原带revision的设置入口把活动对话设为2，沿原队列至少观察每个健康账号一对真实重叠且最终保存成功的请求。保留当前image上限和发送间隔。按小时报告已成功图片、仍未知结果、失败率与等待p95；样本不足或上游限流增加时维持/退回原设置，不推定4路稳定。
3. **四路只作后续实测选择：** 两路稳定且有足够原业务样本后才用同一入口做有界四路试验，比较吞吐和等待、429与UNKNOWN；若无实质收益或错误升高，恢复试验前revision对应数值。配置变更不终止原任务，不释放UNKNOWN。
4. **恢复验证：** 先用隔离回归覆盖无发送领取过期、已发送进程消失、下载失败和前驱失败。线上仅在发布协调的安全窗口验证单服务重启及单账号停用/恢复；保留原请求读回，不断开全公司网络或清库。真实故障事件按实际是否发生单列，不能把测试注入写成真实上游事故。
5. **连续24小时：** 从上述准确部署和原负载就绪时起保留完整UTC起止窗口、各小时日志与资源读回，核对接收数=排队+执行/恢复+已完成/明确失败，抽查原身份和成果无丢失、无重复。没有业务输入时记录空闲原因，不计成24小时持续成功生图。服务重启/账号恢复后仍原ID接续才关闭相应项。

复用 `scripts/audit_pacing_log.py`：对有界完整日志导出运行 `python scripts/audit_pacing_log.py < provider-window.log > pool-window-report.json`。新增 `task_finished` 来自原回执终态，按请求去重并以最新恢复结果统计 `hourly_outcomes`，成功、失败、UNKNOWN分别列出；图片数量证据缺失单列，失败率分母只含已知终态。采样上限只截取样本列表，小时统计遍历全部导入事件。脚本保留 `real_24h_acceptance=unverified`，日志片段/假时钟/短测不能自动关闭验收。无需新定时器或第二套监控服务。

### 原任务持久化、领取与恢复

沿用 `data/text_tasks.sqlite3` 的原 `requests` 表，原图片 JSON 收据一次导入同库 `image_requests`，原 JSON 保留为迁移前历史。恢复所需实际文本/参考图 bytes、原请求身份和必要转发字段写入同目录 `text_tasks_inputs/`（目录0700，文件0600，受控文件名、无符号链接），先 fsync 输入，再提交原 owner/ID/输入一致性记录。公开回执不暴露私密输入、路径或账号凭据；重复同 ID 同输入只读原记录，不同输入409。

领取使用 SQLite `BEGIN IMMEDIATE`：同时核对所有原任务占用、账号状态、顺序，保存原账号绑定与30秒领取租约；执行中每10秒续租。账号 JSON 及完整模型响应流节拍使用进程间文件锁。所有进程必须共享同一本机数据根、同一版本及支持 POSIX 锁的文件系统；验证覆盖本机多进程，不宣称跨主机/NFS或其他账号存储后端已验证。

准入保留已有 `provider_account_identity`。尚无此图片/会话身份字段的合法账号（包括旧、新 Codex-only）使用原 `managed_pool_account_ref` 派生同一个稳定、不含凭据的身份，正常读取不回写账号；以后创建普通会话绑定时保存的也是同一身份。原 pool ref 随既有凭据轮换持久保留，多进程读取沿它接续令牌别名；原 Codex affinity、response 归属及 UNKNOWN 不改绑，不要求生产手工补字段或先做一次图片调用。

模型 POST 前持久记录“可能已发送”并再次检查原领取。租约过期而尚未开始发送的任务重新等待；可能已发送的任务转原 UNKNOWN，不能换号补发。公共持久 `/api/chat-requests`、内部持久 `/api/conversation-bindings/text` 及 `/api/image-tasks` 按原账号、会话及请求消息查询。持久图片已保存结果 ID 时仅重新下载，不调用生成。原生 Codex 已落盘的原始输出可按同请求 ID重新订阅，缺少权威原结果读取能力的 Codex UNKNOWN 保留待查，不发起替代请求。无稳定请求ID的旧客户端不能获得跨客户端重试去重保证，应采用更新后的调用合同。

既有 PPT/PSD 文件任务也经同一原文本任务存储准入，原文件任务查询接口投影同一收据；等待输入重启可恢复，旧文件 JSON 记录不重建。文件任务 UNKNOWN 没有新增自动导出/重生成路径，也不冒称支持图片任务的原结果下载恢复。搜索以其实际固定模型检查可用性。

旧版原图片收据若没有可恢复输入或原请求消息边界，保留旧记录/明确未知状态，不从 hash、prompt相似度或会话末尾猜测重建。兼容接口的多图调用顺序运行显式槽位，已发送槽位不能二次发送；公开持久图片任务仍为一任务一图。

直接原文字 cursor GET 的 TLS 建连失败仍最多重连一次，保留同账号、同会话、同 cursor 与原 60 秒总期限。仅当原生计数完整证明单连接、无 HTTP 请求/响应/上传/下载/early data/重定向且 TLS 未完成时，第二次沿用已扣读取名额，不再等待一个补充周期；账号和读取冷却、共享 HTTP 间隔及读取并行上限仍生效。计数缺失或有歧义时保持原有节流；429、认证和证书失败不走此恢复。它不重发生成、不改变 UNKNOWN、不用于归档或持久任务自动恢复。原始连接失败与后继成功分别保留，恢复成功不代表该轮传输无故障。

直接兼容 `/v1/chat/completions`、`/v1/responses`、`/v1/messages` 的普通文字路径现在保留原 Chat 历史、实际发送根节点、原用户消息 ID、流中返回的会话 ID 和原协议响应 ID。中途 EOF 不再生成成功结束事件；重启后沿同一原回执的恢复租约做有界原账号 GET，唯一完成结果通过原协议格式化并落入新的私密完整响应文件，再原子更新该回执。已打开的订阅固定读取原前缀文件，不把新完整响应拼到旧前缀后；同 ID 重连取得完整结果，原响应/工具调用 ID 保留。恢复不调用生成处理器，不重复写一次调用日志。恢复次序复用原 retry 时间，避免短退避的旧扫描长期挤压其他到期读取。没有完整定位和格式元数据的旧兼容收据仍不进入这个文字恢复器。

`/v1/search` 及 Chat/Responses 的搜索工具路径也沿上述原回执恢复：准备和模型提交使用同一个已持久化用户消息 ID，流中取得的会话 ID 立即保存。没有收到会话 ID 时沿原账号做有界消息定位；只采用原请求唯一终结分支的答案和引用，不取会话最新回答、不以片段稳定或超时判成功。恢复保留原 Responses 搜索调用 ID、消息 ID 和响应 ID，复用同一私密完整输出发布方式。准入检查搜索实际发送的固定模型 `gpt-5-5`，实际模型只投影到原回执的调度字段，原输入封装和哈希保留客户端别名；升级后同ID同输入可继续读旧成功/UNKNOWN结果，不因内部模型映射收到409。启动时校验已保存输入后，只修正尚未领取的旧排队搜索的调度模型，并以原回执比较更新防止覆盖并发领取；该模型额度恢复后原等待请求可重新领取。实际输入变化仍冲突。此项有隔离测试，尚未做真实上游断线恢复验收。

未走公开持久图片适配的 Chat 路线 `/v1/images/generations`、`/v1/images/edits`、Chat 图片和 Responses 图片现在也使用原文本回执的恢复租约。执行前保存格式参数和每图原消息 ID，执行中保存原会话、结果 ID 及私密成果文件；SQLite 备份同时保留这些文件。恢复只读原账号的原分支，已有结果 ID 时直接解析原下载地址，任何原地址失败都不把部分图片报成功。原分支已终结而下载失败时可释放模型轮次，图片生命周期与原结果继续等待；仅有图片文件而无终结证据时继续保留轮次。下载成功成果不重画，后续恢复复用本地文件。

多图中途失败后，先由同一恢复租约确认并保存已发送图，再将原回执的未发送余项放回原准入调度；账号恢复后原请求自动续行，仍绑定原账号并参与来源轮转。过期恢复 worker 不能覆盖新领取结果。完整协议响应从保存的成果重新整理，保留已发出的响应 ID；旧订阅只读原前缀，同 ID 重连取得新的完整响应。恢复记录区分读取原会话、解析下载地址、下载、保存和格式化阶段；ChatGPT GET 的429保留原请求关联及账号冷却证据，对象存储下载429仅退避下载、不使ChatGPT账号冷却。以上为隔离测试证据，尚未做真实上游图片断线/重启验收。

尚未完成的 P3 范围：原生 Codex UNKNOWN 的完整上游结果读取。原生 Codex 在 `response.created` 时保存真实上游 response ID 和可得 HTTP 请求 ID，但没有凭此虚构可用的 GET 恢复端点。原生 Codex UNKNOWN、缺少定位/格式元数据的旧 raw 图片收据仍只重读已落盘响应或前缀，不补造尾部、不重发；不能称所有兼容路径的原结果恢复已完成。带公开图片标记的同步图片接口继续沿 `/api/image-tasks` 原收据查询和恢复下载。

结果 UNKNOWN 与模型轮次是否结束分开计算：原 GET 只有在原用户的唯一完整分支上证明最终助手消息 `finished_successfully/end_turn=true`、且没有活动或歧义分支时，才可保存轮次结束证据；空结果仍保持 UNKNOWN 及同会话顺序，其他独立等待可使用已经证实释放的轮次。旧公开 Chat 收据只有明确 `stream_open/http/conversation/422` 的原模型拒绝证据可直接排除模型占用，状态和原输入身份不改写。原生持久 Chat 文字请求在本地执行者已退出或原执行领取已过期、且没有恢复读取者持有领取时，沿已有规则处理：受理至少15分钟，并在此后取得至少3次符合条件的原结果查询，最近一次仍为允许的无结果原因，自动保存 `failed/RESULT_UNRECOVERABLE` 与私有 `_execution_wait_ended_at`，同时撤销过期领取以阻止旧worker续租、回写或再次发送，释放本地调度位。后台查询与显式恢复采用相同计数；已有合格证据在下一次原子调度时同样生效。这是结束本地执行等待，不能证明上游终结：`upstream_outcome=unknown`、原账号/会话/消息、同会话前驱保护及原ID查询均保留，不重发原请求。普通超时、网络/授权/限流查询失败、活动或歧义分支、次数不足，不能单独触发此决定。迟到的准确原结果仍可保存；后续读取失败或重启不能重新占回已释放的位。已有明确空终态保持原纠正合同。此规则不扩展到原生Codex、兼容转发、文件或图片任务；图片生命周期占用仍另算。

缺少原上游对话定位时，恢复扫描只在原账号内按原用户消息 ID 查找。单个候选的404、超时、5xx或格式错误保留为未读证据，并让同一有界窗口中的其他候选继续检查；每个候选每轮最多读取一次。候选重试从30秒开始退避，服从更长 Retry-After；账号级401/403/429立即结束本轮，不改号绕过。原20秒扫描预算、100候选持久窗口和SQLite领取不变。窗口满且全部候选不可读取时仍明确受阻，不丢弃未读项以扩展范围，也不据此判定没有结果。

`_recovery_conversation_scan.failed_reads` 是原私密回执内的增量字段，保存未读候选及脱敏错误、次数、下次时间；成功检查的候选不会因另一个失败被反复重查。跨页保留未读项与已找到的暂定匹配。列表覆盖完成不等于详情已全部检查；仍有未读项时，即使已找到一个匹配也不能宣布唯一结果、修改原上游绑定或释放占用。全部覆盖和详情取证完成后，仍使用原精确父消息及唯一终态分支验证。旧扫描记录可继续读取；不自动重放旧输入，不解除显式 `_recovery_suppressed` 停止标记。

公共持久Chat回执的 `recovery.scan` 分开展示 `list_complete`、`listed_total`、当前窗口候选/已检查/未读数量、暂定匹配数，以及脱敏候选失败。`recovery.last_read_error` 保留阶段、HTTP状态、错误分类和Retry-After，不保存异常正文、地址、标题、凭据或原账号/对话ID。候选引用仅为稳定哈希，操作人可在授权的原环境内核对；恢复次数仍不等于完整扫描次数。`recovery.next_at` 是任务级下一次恢复时间，单个失败候选的 `next_at` 是其最早可重查时间。

无需触发恢复GET即可使用 `python scripts/audit_recovery_scan.py --database <授权SQLite路径> --owner <原归属> --request-id <原请求编号>` 检查一条原回执。脚本只用SQLite `mode=ro`/`query_only` 读取，不导入Provider、不访问上游、不创建缺失数据库、不修改回执或停止标记。使用一致的授权副本或既有可读SQLite/WAL文件，不用 `immutable=1` 忽略最新WAL。报告只含经过筛选的状态/计数及候选哈希；它不是结果恢复成功、未执行证明或重发许可。生产旧请求是否恢复、操作人停止的记录是否继续及新的业务验证均另按原授权取证。

本次真实只读范围（2026-09-21）：现役仍为 Provider42，候选基线 `a8697d8036edf9e8c34815147a772cfa785b40d1`。原 `inv-g8wggugvrj` 对9个指定 UNKNOWN 最多12次 GET、0 POST，9条原回执字节不变，仍不能归属原结果；账号文件同期有变化，不能将其宣称为未变或归因于本读取。后续 P6 新原请求使现场成为512文本/10 UNKNOWN；`inv-b8wh180jme` 仅查询这10个原ID的白名单诊断，0 Provider/GET/POST/写。其中2条有完整模型入口422证据，2条422缺阶段、2条timeout/stream_open、4条缺诊断。因此此代码对这组证据只能排除2条模型占用，其余8条继续未知，不能宣称已解除生产迁移容量阻塞。这10条是status=unknown样本；精确副本预检还须把旧failed/RESULT_UNRECOVERABLE且upstream_outcome=unknown的记录计入，不能将这10条当作全部未知执行。原精确制品预检 `inv-d8wfs006gr` 的236图/217成功/509文本保存证据与Chat可派发0的阻塞保留；本轮没有生产切换、原任务重发或新图生成。

协议在发送前明确拒绝（如空消息HTTP400）保存 `failed/PROTOCOL_REQUEST_REJECTED` 和原HTTP错误语义，不变成UNKNOWN或无限重排。只有明确的可恢复容量/领取暂不可用才等待。兼容执行者通过原 `LoggedCall` 在结果落盘或失败/流结束时记录一次调用，保留Key、原请求ID、实际账号及可得429证据；HTTP订阅断开不取消记录，同ID重连不重复写成功日志。

部署时须由协调者按 Workbench 当前 runbook 和不可变 digest，先完成候选制品预检与可恢复备份，停止旧 Provider 写入者后再导入/启动，不能让仍写 `image_tasks.json` 的旧版本和新版本混跑。现有备份的 `image_tasks` 选项包含一致的 SQLite 副本和所有引用的私密输入、已提交输出前缀；它们应与原账号、节拍、图片存储整体保护。回退保留切换后新增/修改的 SQLite 记录与输入，不得直接用旧 JSON 覆盖回退或删 UNKNOWN 后重画；旧版本不理解新等待记录，故必须先停止新写入者，逐项核对待处理记录和恢复路径。此次没有执行迁移、部署或生产参数调整。结束本地等待的新字段保存在原SQLite收据内；回退到不识别该字段的旧版会重新保守计入这些未确认轮次。回退预检必须核对这一容量变化，不能删除原收据或把上游结果改为未发送。

内部 bound-text 与公共顺序会话共用原 Chat 的发送前检查：以原账号读取原 Chat，核对会话身份、当前父消息和明确的归档状态；已归档时复用既有取消归档及游标读回，确认后才发送模型请求。读取/恢复不确定时沿原任务保存 `CHAT_ARCHIVE_RESTORE_UNCONFIRMED`、`upstream_outcome=not_sent`；接入准入调度的任务继续等待，不新建 Chat 或切换账号。该检查只作用于获准执行的新轮次，不重排旧 `UNKNOWN` 或 `failed/RESULT_UNRECOVERABLE`，也不把前一轮父消息结束当作本轮结果。此修复不授权部署时批量取消归档或重发历史请求。

### 内部缺失原消息的显式后继

此能力只属于现有 admin/Content `POST /api/conversation-bindings/text`；普通 Key、公共 Chat 和 Happy 合同不变。默认 UNKNOWN 仍只查原结果。只有业务明确决定承担潜在重复风险后，才可提交以下七个字段（示例是虚构身份）：

```json
{
  "supersedes_request_id": "old-request",
  "client_request_id": "one-persisted-successor",
  "provider_binding_id": "original-binding",
  "provider_account_identity": "original-account",
  "client_conversation_id": "original-client-session",
  "conversation_id": "original-chat",
  "parent_message_id": "original-submission-parent"
}
```

此分支禁止携带 `messages`、`model`、`image_model`、`thinking_effort` 或其他字段（下述受限目录派生参数除外）；包括传入默认值也会拒绝。服务在同 owner 下读取原 `_input_ref` 并核对原请求 hash、绑定和提交父消息，复用完整原输入，仅替换新请求 ID、附旧引用，并在受理前持久保存新请求的完整输入。不会向客户端返回提示词或私密文件引用。普通请求省略新字段时，原默认值及 request hash 语义不变。

前提是原请求为内部 Chat 文本、`failed/RESULT_UNRECOVERABLE`、`upstream_outcome=unknown`、本地执行等待已结束、原消息缺失，且无有效执行/恢复领取或停止标记。原 SQLite `BEGIN IMMEDIATE` 中复核，一个原请求最多一个后继；原记录、消息 ID 和发送次数不修改。不允许借此跳过同会话的其他未完成请求，也不支持递归替换后继 UNKNOWN。

| 情形 | 返回及调用方动作 |
| --- | --- |
| 同新 ID、相同七字段 | 返回原新回执，继续查此 ID，不再次生成 |
| 同新 ID 改字段 | HTTP 409 `CONVERSATION_REQUEST_CONFLICT` |
| 不同新 ID 争同一旧 ID，或存在其他未完成轮次 | HTTP 409 `CHAT_SUPERSEDE_CONFLICT` |
| 原输入缺失/损坏、身份不同或旧状态不符合 | HTTP 409 `CHAT_SUPERSEDE_INVALID`，未受理 |
| 原执行/恢复仍有有效领取 | HTTP 409 `CHAT_SUPERSEDE_PREDECESSOR_BUSY`，未受理；等待后沿同新 ID 核验 |
| 已受理后查到旧 user 或旧终态 | 新回执 `failed/not_sent`、`CHAT_SUPERSEDE_ORIGINAL_FOUND`；查 `supersedes_request_id` 的原结果，不换新 ID 再发 |
| 原 parent/分支漂移 | 新回执 `failed/not_sent`、`CHAT_SUPERSEDE_CURSOR_CHANGED`，停下处理原会话 |
| 原 Chat 读失败/限流，或原领取暂忙 | 新回执 `queued/not_sent`、`CHAT_SUPERSEDE_READ_UNAVAILABLE` 或 `CHAT_SUPERSEDE_PREDECESSOR_BUSY`；保存同一新 ID 等待 |

响应新增 `supersedes_request_id`，结果只按新请求自己的 user/assistant 分支核验。领取及发送前都会重核原记录；原 Chat GET 在账号冷却后执行，早于模型提交标记，GET 同样计入发送间隔并遵守真实 429 冷却。原账号、Chat、提交父消息、客户端会话固定；已归档 Chat 复用原取消归档及读回检查。

**这仍然是新执行，不是取回旧结果。** 最后一次 GET 后仍须遵守账号请求间隔，上游也没有原子“确认旧请求不存在并提交新请求”的接口；GET 与 POST 之间仍可能出现旧结果。工程只能阻止已观察到的迟到结果和本地重复派发，不能承诺上游 exactly-once。本候选测试不授权历史请求生产重发；业务执行必须另核对应旧对象及重复风险决策。

无数据库迁移、新服务或配置变化。发布需先 Provider、后使用此字段的内部消费者。接受后继之前可回退旧镜像；一旦已受理后继，**不得直接降级为不识别此合同的版本**：应保留识别此合同的运行版本处理在途/等待及原 ID 读回，或由发布负责人核定只暂停新提交的回退方案，保留原数据库和输入。不能删除后继或清空 UNKNOWN 来回退。

#### 已发送但无最终回答的显式原会话后继

同一内部接口的七个身份字段可额外带 `continue_after_no_final: true`。此模式仅用于旧公司原生文字请求已经以 `ORIGINAL_RESULT_NO_FINAL` 停止自动恢复、至少三次有效读取仍为 `REQUEST_RESULT_NOT_FOUND`，且无工作生命周期、completion、可重试游标、暂停或活跃领取的情况；客户端不能替换提示词、模型或身份。只有下面的 `attributes_required_only_v4` 封闭升级可与此标志合用；旧 `category_directory_parent_v1` 仍不可合用。默认省略时继续使用原消息缺失分支，不自动开启重试。

Provider 保存唯一幂等后继，复用原输入、原账号和原 Chat。worker 首次读取必须证明原 user 后只有已完成的 assistant code / tool 唯一链，没有最终回答；发送前再次读取，要求尾节点和整条链的内容、角色、完成状态一致。迟到 final、分叉、新 user、运行中的步骤、归档或内容变化均拒发。实际发送父节点是已观察的 tool 尾节点，原提交父节点和旧 UNKNOWN 收据不改写；新回执只认新请求的回答。两次读取之间的内容比较只存在内存，不保存私密上下文副本。此模式仍有 GET/POST 间上游迟到的风险，不等同上游取消或 exactly-once。

#### 旧属性任务的限定 v4 / High 升级

同一内部入口只接受七身份字段、`continue_after_no_final:true` 和
`derived_input:{"kind":"attributes_required_only_v4"}`。不能带新 messages、model、thinking_effort、目标类别或商品资料。
原输入须为 `gpt-5-6-instant/standard`、字面一致的已知旧类目或 required-only-v1 模板；类目模板必须只有一个可完整解码的目录叶，即使重复 ID 也不去重选取。属性模板的固定类别、required targets、schema 必须一致。未知前缀、额外指令、重复 JSON 字段、损坏的 image_ref/图片配对、schema 或来源属性索引均拒绝，不猜测修补。

Provider 从保留输入确定性生成 required-only-v4，固定 `gpt-5-6-thinking/high`；全部 SOURCE 值、来源属性原序、image_ref 与图片部分保持不变。schema 仅筛 `required && !provided_by_category && !already_satisfied`，每个入选行的字典及其他元数据完整保留；完整旧 schema/输入仍保留在原私密 input，旧 hash/回执/UNKNOWN 不改。v4 明确引用本请求真实 image_ref，要求正向商品证据，不默认猜测人群。同一账号不支持目标模型时拒发，不能换号或另开会话。原 no-final 资格、唯一后继、重启幂等、发送前原记录及两次上游分支校验全部复用。

回执返回 `continue_after_no_final:true`、`supersedes_request_id` 和 `derived_input` proof：`kind`、`original_input_hash`、`source_contract`（`legacy_category` / `required_only_v1`）、`original_model`、`original_thinking_effort`、`model`、`thinking_effort`、`target:{description_category_id,type_id}`、`schema_count`、`required_target_count`、`image_count`、`text_utf8_bytes`。不返回提示词、图片或 private input 路径。`model`/effort 升级只允许此 kind，派生 body/hash/proof 在受理和真正发送前重算核对。

输入不受支持、非法组合或超 64KiB 文本分别返回 `CHAT_DERIVED_INPUT_UNSUPPORTED/INVALID/TOO_LARGE`；原件资格、活跃领取、原结果出现、第二后继及 ID 漂移仍用现有 `CHAT_SUPERSEDE_*` / `CONVERSATION_REQUEST_CONFLICT`。上游读取暂时失败的已受理后继保留同 ID `queued/not_sent`，不能凭此再建一个。此 mode 不适用旧类目 `resume-unsent-successor`。Provider 先发布，消费者再启用；一旦接受此种后继，回退版本必须仍识别其持久输入和关系。

#### 413 类目目录的受限派生输入

上述七字段可额外带 `derived_input: {"kind":"category_directory_parent_v1"}`。仍禁止客户端提供替换 messages。除普通后继的全部前提外，原回执必须保留 `stream_open/http/conversation`、HTTP 413 和已开始提交的证据；413 本身不证明未发送，原 UNKNOWN 不变。

Provider 只识别保留的 Workbench 类目专用提示词：单 user 消息、空 schema、明确类目推荐指令，目录为 `compact_directory` 五元组或 `directory_candidates` 对象数组，所有叶必须有完整非空路径。未知格式、属性任务、缺层级拒绝为 `CHAT_DERIVED_INPUT_UNSUPPORTED`；参数或413证据不符为 `CHAT_DERIVED_INPUT_INVALID`。转换保留全部 SOURCE 字段、图片及 image_ref，以原顺序和重复叶计算每个真实根节点的类别数量，不做语义筛选。完整序列化文本（含消息封装和 image_ref，图片数据沿独立上传路径）UTF-8 超过 65536 字节时拒绝为 `CHAT_DERIVED_INPUT_TOO_LARGE`。

派生结果仅允许类目导航：`{"determined":true,"selection":{"kind":"branch","path":["真实根名称"],"scope":"subtree"},"attributes":[]}`，或 `{"determined":false,"attributes":[]}`。消费者必须验证该分支确实在原完整目录中；分支不是最终类目，不能直接用于发布。后续子目录沿该后继成功的实际游标继续。原请求若恢复成功，优先处理原结果，不再发送派生请求。

新回执的 `derived_input` 返回 `kind`、`original_input_hash`、`directory_sha256`、`category_count` 和 `text_utf8_bytes`。原 hash 由 Provider 从原 DB 记录绑定，无须客户端提供。目录指纹为 SHA256(UTF8(JSON.stringify(rows)))，其中 rows 按原序列保留全部重复叶，每行为 `[descriptionCategoryId,typeId,name,descriptionCategoryName ?? null,categoryPath]`；消费者可从冻结 recovery_request 重算并对照。派生请求保存自己的输入和 hash，并复用普通后继的唯一性、幂等及发送前原分支检查；旧输入及 hash 不改写。

#### 原类目后继已证实未发送时的一次性恢复

仅 admin/Content 可显式调用 `POST /api/conversation-bindings/text-requests/{request_id}/resume-unsent-successor`，body 是该后继原七身份字段及 `derived_input:{"kind":"category_directory_parent_v1"}`，URL ID 必须与 `client_request_id` 相同。禁止替换 messages、账号、会话、父节点或原请求引用；不新建后继 ID 或授权记录。运行实例必须具备持久 admission；缺失时 GET 不显示可恢复标志，显式接口拒绝且不消费一次性重排额度。

本地校验要求原后继为 `failed/CHAT_SUPERSEDE_CURSOR_CHANGED/not_sent`、提交/占位/执行标记明确为 false、没有发送序列或发送时间线、无有效 claim/人工暂停、保留派生输入及双方 hash/绑定全部相符。GET 仅在全部本地校验通过时返回 `bound_successor_resume_retryable:true`、`upstream_outcome:not_submitted`、`upstream_submission_started:false`；此标志不证明上游父节点当前可用。GET、普通 submit 和 recover 均不重排。

显式调用在原 SQLite 事务中最多重排一次，保留原 request/message ID、input/hash、唯一后继关系和失败快照。并发或重启后的同 ID 同参数调用只返回同一回执；恢复后再次失败不会获得第二次重排。资格不足返回 HTTP409 `CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE`，身份变化仍返回原冲突错误。

正常 runner 继续在初次读取和账号节拍后的发送边界检查原请求/父节点。除原 assistant 终态外，仅兼容已完成 user → 已完成 assistant code 调用 → 已完成 image tool 的唯一父链：调用 recipient 必须等于 tool author.name，tool 必须有合法 `image_asset_pointer` 且无子节点。任意普通 tool/null end_turn、活跃状态、缺失链条、兄弟分支、实际 current node 变化或旧请求出现均不能因此绕过检查。


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
