# 上游 Conversation SSE 协议说明

Conversation SSE 是上游对话链路的流式返回协议。每条 SSE `data:` 通常是一段 JSON payload，也可能是协议标记或结束标记。客户端需要按顺序消费这些 payload，维护当前会话状态、文本内容、工具调用状态和图片结果指针。

## 基本形态

常见 payload 示例：

```text
"v1"
{"type":"resume_conversation_token",...}
{"p":"","o":"add","v":{...}}
{"v":{...}}
{"p":"/message/content/parts/0","o":"append","v":"..."}
{"type":"server_ste_metadata","metadata":{...}}
[DONE]
```

处理建议：

| payload | 含义 | 处理方式 |
|:--|:--|:--|
| `"v1"` | 协议版本标记 | 可记录，通常不影响业务 |
| `[DONE]` | 当前 SSE 流结束 | 停止继续读取 |
| JSON object | 事件、消息或 patch | 按字段更新会话状态 |
| JSON string | 短文本 patch 或协议标记 | 结合上下文处理 |
| 非 JSON 内容 | 原始内容 | 保留为 raw 事件，避免中断流 |

## 常用字段

| 字段 | 说明 |
|:--|:--|
| `type` | 上游事件类型，如 `resume_conversation_token`、`input_message`、`message_marker`、`title_generation`、`server_ste_metadata` |
| `conversation_id` | 当前会话 ID，可从多个事件中获得 |
| `p` | patch 路径，例如 `/message/content/parts/0` |
| `o` | patch 操作，例如 `add`、`append`、`replace`、`patch` |
| `v` | patch 值，可能是字符串、数组，也可能包含完整 message |
| `c` | 消息序号或游标，常见于 add 类事件 |
| `message.id` | 消息 ID |
| `message.author.role` | 消息角色，常见 `system`、`user`、`assistant`、`tool` |
| `message.content.content_type` | 内容类型，如 `text`、`multimodal_text`、`model_editable_context` |
| `message.content.parts` | 内容片段，可能包含文本、图片指针或多模态对象 |
| `message.status` | 消息状态，如 `in_progress`、`finished_successfully` |
| `message.end_turn` | 是否结束当前轮次 |
| `metadata.tool_invoked` | 本轮是否调用工具 |
| `metadata.turn_use_case` | 本轮用途，如 `text`、`multimodal` |
| `metadata.async_task_type` | 异步工具任务类型，图片生成通常为 `image_gen` |

## 会话启动事件

上游通常会先返回恢复令牌或会话令牌：

```json
{
  "type": "resume_conversation_token",
  "kind": "topic",
  "token": "...",
  "conversation_id": "..."
}
```

这个事件主要用于标识会话和恢复上下文。业务层通常只需要保存 `conversation_id`，`token` 不应该暴露给下游用户。

## 消息 add 场景

完整消息可能通过 `add` 或带 `v.message` 的事件出现：

```json
{
  "p": "",
  "o": "add",
  "v": {
    "message": {
      "author": {"role": "assistant"},
      "content": {"content_type": "text", "parts": [""]},
      "status": "in_progress"
    },
    "conversation_id": "..."
  },
  "c": 3
}
```

此类事件常用于创建一条新消息。若消息角色为 `assistant`，后续文本通常会通过 patch 继续追加。

## 文本增量场景

文本输出通常由多条 patch 组成：

```json
{"p":"/message/content/parts/0","o":"append","v":"Hello"}
{"v":" world"}
{"p":"","o":"patch","v":[
  {"p":"/message/content/parts/0","o":"append","v":"!"},
  {"p":"/message/status","o":"replace","v":"finished_successfully"},
  {"p":"/message/end_turn","o":"replace","v":true}
]}
```

处理要点：

| 形态 | 含义 |
|:--|:--|
| `p == "/message/content/parts/0"` 且 `o == "append"` | 向当前文本追加内容 |
| `o == "replace"` | 用新值替换目标字段 |
| `o == "patch"` 且 `v` 是数组 | 批量 patch，需要按数组顺序处理 |
| 只有 `v` 且 `v` 是字符串 | 可能是省略路径的文本增量，应结合当前文本流处理 |

## 输入消息场景

用户输入会以 `input_message` 或普通 `user` message 出现。图片编辑请求会包含用户上传的参考图：

```json
{
  "type": "input_message",
  "input_message": {
    "author": {"role": "user"},
    "content": {
      "content_type": "multimodal_text",
      "parts": [
        {"asset_pointer": "sediment://file_input"},
        "编辑提示词"
      ]
    }
  },
  "conversation_id": "..."
}
```

这类 `sediment://...` 表示输入附件，不是生成结果。即使它可以被下载，也不能当作输出图片返回。

## 图片工具成功场景

图片生成或图片编辑成功时，上游一般会出现工具消息：

```json
{
  "v": {
    "message": {
      "author": {"role": "tool"},
      "content": {
        "content_type": "multimodal_text",
        "parts": [
          {"asset_pointer": "file-service://file_result"},
          {"asset_pointer": "sediment://file_result"}
        ]
      },
      "metadata": {"async_task_type": "image_gen"}
    }
  },
  "conversation_id": "..."
}
```

只有同时满足以下条件的图片指针，才应该视为输出结果：

| 条件 | 说明 |
|:--|:--|
| `message.author.role == "tool"` | 来源是工具消息 |
| `metadata.async_task_type == "image_gen"` | 工具任务是图片生成 |
| `asset_pointer` 为 `file-service://...` 或 `sediment://...` | 指向可解析图片资源 |

## 图片指针类型

| 指针 | 常见来源 | 说明 |
|:--|:--|:--|
| `file-service://file_xxx` | 图片工具输出 | 可通过文件下载接口解析 |
| `sediment://file_xxx` | 输入附件或图片工具输出 | 需要结合消息角色判断来源 |
| `file_upload` | 上传过程占位 | 通常不应作为输出 |

不要只凭字符串里出现 `file_` 或 `sediment://` 就判定为输出图。必须结合消息角色和任务类型。

## 策略拒绝场景

当上游拒绝请求时，通常不会产生图片工具消息，而是返回普通 assistant 文本：

```text
I can't assist with that request. If you have another type of modification...
```

常见伴随事件：

```json
{"type":"title_generation","title":"Request Denied","conversation_id":"..."}
```

```json
{
  "type": "server_ste_metadata",
  "metadata": {
    "tool_invoked": false,
    "turn_use_case": "multimodal",
    "did_prompt_contain_image": true
  },
  "conversation_id": "..."
}
```

处理要点：

| 条件 | 行为 |
|:--|:--|
| 有 assistant 拒绝文本 | 应返回文本消息 |
| `tool_invoked == false` | 说明没有实际工具结果 |
| 没有 `role=tool` 且 `async_task_type=image_gen` 的消息 | 不应收集输出图片 |
| 用户输入消息里有图片指针 | 仍然只视为输入附件 |

## moderation 场景

部分请求可能返回 moderation 事件：

```json
{
  "type": "moderation",
  "moderation_response": {
    "blocked": true
  },
  "conversation_id": "..."
}
```

若 `blocked == true`，应认为本轮被策略拦截。后续如有 assistant 文本，应优先返回该文本；若没有文本，可返回合适的错误信息。

## marker 和 title 事件

上游会返回一些辅助事件：

```json
{"type":"message_marker","marker":"user_visible_token","event":"first"}
{"type":"message_marker","marker":"last_token","event":"last"}
{"type":"title_generation","title":"...","conversation_id":"..."}
```

这些事件通常用于前端展示、标题生成或流式状态标记，不代表实际文本内容或图片结果。

## metadata 事件

`server_ste_metadata` 用于描述本轮调度和工具状态：

```json
{
  "type": "server_ste_metadata",
  "metadata": {
    "tool_invoked": true,
    "turn_use_case": "multimodal",
    "model_slug": "i-mini-m",
    "did_prompt_contain_image": true
  }
}
```

常用判断：

| 字段 | 说明 |
|:--|:--|
| `tool_invoked == true` | 上游认为本轮调用过工具 |
| `tool_invoked == false` | 上游未调用工具，常见于拒绝或纯文本响应 |
| `turn_use_case == "text"` | 按文本响应处理 |
| `turn_use_case == "multimodal"` | 多模态请求，不代表一定有图片输出 |
| `did_prompt_contain_image == true` | 输入包含图片，不代表输出包含图片 |

## 结束后的结果判断

SSE 结束后可按以下顺序判断结果：

1. 如果已经收集到图片工具输出指针，解析并下载输出图片。
2. 如果没有输出图片指针，但有 assistant 文本，并且本轮被拦截或未调用工具，返回文本消息。
3. 如果没有输出图片指针，但有 `conversation_id`，可查询完整会话明细，继续寻找图片工具输出。
4. 查询完整会话时，仍然只读取 `role=tool` 且 `async_task_type=image_gen` 的消息。
5. 如果没有图片结果也没有文本，返回上游异常或空结果错误。

## 初始文案请求结果不明时的只读恢复

`GET /api/conversation-bindings/text` 复用现有身份认证，要求传入原请求保存的
`provider_binding_id`、`provider_account_identity`、`client_conversation_id`、
`conversation_id` 和 `parent_message_id`。只在原绑定账号查询原会话，不发送消息、
不切换账号、不创建新绑定。调用方必须保存原响应中的这些范围字段；不能由商品编号
或时间推测、重建已经丢失的会话引用。

只有原锚点仍在当前分支、没有后续用户消息，且当前消息是已结束的完整 assistant
正文，才返回 `status=succeeded`、`binding_status=bound` 和正文。分析文本、未结束
或空正文返回 `status=running`，不能当作生成成功。账号或会话错配返回 409。

初始文本流异常且已取得会话编号时，服务先查询一次既有结果；未完成则保留原引用供
后续 GET 查询，不重复生成。该自动恢复不用于后续聊天轮次，避免取到前一轮回答。
Workbench 消费者需先部署此 Provider 接口，再启用保存和查询原文案引用的 Worker；
历史记录缺少原引用时仍是结果未知，不能自动重提。

## 独立控制原会话读取节奏

生成发送、附件准备和查询已生成结果是不同的负载。管理员可以通过既有
`POST /api/settings` 单独设置会话 GET 的节奏；`GET /api/settings` 返回实际生效值。
这些值在服务实例内统一配置，但额度、冷却和在途计数按账号隔离，不是上游官方配额。

| 设置 | 默认值 | 含义 |
|:--|--:|:--|
| `account_request_interval_secs` | 5 | 同账号所有受控 HTTP 的最小间隔；已有显式值保持 |
| `account_message_interval_secs` | 30 | 同账号新消息间隔；通过设置接口明确写入 0 才关闭此间隔 |
| `account_conversation_read_interval_secs` | 0 | 每补充一条原会话读取额度需要的秒数；0 保持旧共用 HTTP 节奏 |
| `account_conversation_read_burst` | 1 | 最多积累多少条读取额度；整数 1–100 |
| `account_conversation_read_max_inflight` | 0 | 同账号正在执行的原会话 GET 上限；整数 0–100，0 保持旧的不限此项行为 |

例如 R=45、B=4、在途上限=4，表示自然空闲后最多积累四次读取，消耗后每 45 秒
补一条，并且实际在途读取不超过四条。它不表示“每 45 秒任意读取四次”。
额度存入既有账号读取时钟，重启或改配置不重置时钟，也不会凭空补满额度。
原 FIFO、取消、期限、归档顺序和 429 冷却继续约束请求；出现限流后突发额度退为单条。

### 已有有限混合样本对应的配置

2026-10-11 隔离候选 `1349276c616f66aec6d9788ad27bde02284eb04c` 在 Nicolas 上
完成一批 4 文字和 2 图片：6 个原请求各发送一次，6 个结果保存，6 个归档确认，
65 次账号 HTTP 均 200、无重试。全部保存约 149.7 秒，全部归档约 638.9 秒。
这是候选范围依据；本次生产兼容移植尚需正式安装后的有限验收，不能直接继承旧候选的实际耗时。

```json
{
  "account_request_interval_secs": 0.25,
  "account_message_interval_secs": 0,
  "account_conversation_read_interval_secs": 45,
  "account_conversation_read_burst": 4,
  "account_conversation_read_max_inflight": 4
}
```

该样本另外设置 Chat 并发 6、图片并发 2，Codex 并发保持 4。并发设置应通过既有
`/api/workbench/ai/pool/resource-settings` 接口携带刚读回的 `expected_revision` 更新；
不能混在 `/api/settings` 保存中，也不能覆盖后来版本。4 文字＋2 图片是样本组成，
不是固定配额或已证明的最大容量。此处记录配置，不自动改变任何生产默认值或账号启用状态。

### 切换与一次有限验收

共享生产服务由现有发布负责人切换。先阻止业务侧新增提交并确认实际在途处理结束，
核对既有队列、普通执行、恢复及归档租约；原 UNKNOWN 保持原请求，只允许原负责人使用
现有明确暂停控制，不能靠清空状态制造可重启窗口。旧版 SIGTERM 不保证等待这些后台线程。
仅凭 active=0 或进程健康不足以安全切换，也不能把新版本的停机改进当作旧版本已有能力。

旧版本的首次切换使用现有控制：提交 Worker 停止接收并实际退出；原负责人对尚未发送的
文字／图片使用各自 `/work` 接口暂停，对可恢复原件使用 `/recovery-control` 暂停并确认
`paused` 且 `in_flight=false`。两种暂停作用不同，`pausing` 也不能视为完成。
已抢到的执行或恢复租约、正在进行的原会话归档须自然结束；队列缺少可暂停的 work 记录
或归档仍为 pending/unknown/running 时，交回对应负责人处理原件。最后的只读核对须覆盖
文字、图片和 work 归档三者，不能仅统计 running 行；切换不清除任何回执或租约。

先按不可变镜像完成代码切换并读取实际设置支持，再应用明确选择的参数。
设置两类接口之间有先后次序：先启用独立读取保护，再提高并发或缩短发送间隔。
每步读回；任何未知响应先查原写入，不能盲目重发。回退只恢复本次保存的配置差异，
保持原账号、回执、结果及读取时钟，不用旧数据库覆盖后来的成功成果。

实际收益用一个有限普通客户端样本核对：受理到发送、结果保存、归档释放的时间，
原请求唯一发送次数、所有错误和恢复记录。已有成功结果不重生成；不要求额外三轮无故障长测。
若参数只在单账号验证，发布负责人须明确部署实例及账号适用范围，不能默认为全池任意账号已验证。
