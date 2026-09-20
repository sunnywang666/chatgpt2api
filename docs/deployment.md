# 部署与升级指南

本文介绍 ChatGPT2API 的常见部署方式，以及后续升级项目时需要保留的数据和执行步骤。

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
