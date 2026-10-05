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

## 图片完成提示与读取兜底

图片等待期间可共享同账号的 `conversations` WebSocket 订阅。只有已订阅且匹配
当前会话的完整图片工具消息才唤醒原请求读取；提示本身不证明图片已保存，也不
绕过账号读取时钟。断连仍使用原查询兜底，不重新生成。
断连后新会话接入时，会建立新的共享监听；旧会话继续原查询兜底，其退出不会
停止新监听。每次新接入至多启动一次连接，仍受既有账号 HTTP 时钟和连接预算
限制，不在等待循环里反复重连。
普通网络错误及未附带 `Retry-After` 的 5xx 退避也可由完成提示提前唤醒；
429 和任何明确 `Retry-After` 仍保留原等待，实际 GET 仍受账号读取时钟限制。
已安装完成监听的图片轮询不在每个空快照后额外读取任务列表，只在最后一个
活跃预算窗口诊断一次；原会话中的内容拒绝仍立即识别，恢复快照不额外查询。
本次 SSE 尚未提供资产时，首次原会话读取优先等待完成提示。当前会话的
assistant/tool `add-messages` 进度会重新开始已有 `image_poll_interval_secs`
安静期；连续一个周期没有进度才查询兜底，避免仍在生成时提前读空。
其他会话、用户消息和标题更新不延长等待。提示到达或监听断开均立即唤醒；
总等待仍受本次 active budget 限制，并为兜底读取保留 10 秒网络预算。
已经取得资产 ID 或已有恢复快照时直接核验，不加首读等待。漏通知时仍有界查询。

`upstream_completion_listener` 日志区分连接、订阅确认、会话注册、提示匹配、
提示消费、断连及退回查询；结束记录汇总订阅前消息、其他主题、未注册会话、
当前会话非终态消息和匹配提示的数量。标识仅使用哈希，不记录签名 URL、消息
正文或异常文本。只有 `hint_matched` 证明接到了可路由提示，`hint_consumed`
证明等待被唤醒；没有信号记录不能直接推断上游没有发送通知。

### 有限突发读取与持续节奏

`account_conversation_read_burst` 默认 `1`，保持原来的逐次读取间隔。
管理员显式设为大于 `1` 时，同账号的结果查询和归档查询共用有限读取次数：
空闲每经过 `account_conversation_read_interval_secs` 补充一次，最多积累到
所设数量；有剩余次数即可按共享 HTTP 节拍发送，用完再等待补充。
它限制所有 conversation GET，不是每个会话各拿一份额度，也不是只给完成通知开后门。
读取间隔为零时不使用该突发限制，仍保留原有 HTTP 节拍与冷却。

剩余次数在已有账号时钟内、HTTP 发送前落盘；失败的传输也占用一次，重启不
重置次数。旧时钟的未来读取时间保留；变更间隔或容量最多继承一次旧可用次数，
不因反复修改设置直接补满。任何活跃的读取或账号 429 退避均退回逐次读取，
`Retry-After`、普通查询 FIFO、归档预约、原会话严格核验保持有效。
调度预判与实际发送共用同一计算，迟到响应不得覆盖其他请求已消耗的次数或冷却。

这是表达有界测试配方的调度能力，不声明任何容量是上游安全上限。
2026-10-05 隔离诊断在 900 秒无原会话查询之后完成 10 次短突发，随后每分钟
一次共 10 次，20 次原 GET 均一次成功；实际突发峰值 10。这一有限样本不能
代替普通客户端持续生图、保存、归档的验证，也未据此修改生产默认参数。

## 初始文案请求结果不明时的只读恢复

`GET /api/conversation-bindings/text` 复用现有身份认证，要求传入原请求保存的
`provider_binding_id`、`provider_account_identity`、`client_conversation_id`、
`conversation_id` 和 `parent_message_id`。只在原绑定账号查询原会话，不发送消息、
不切换账号、不创建新绑定。调用方必须保存原响应中的这些范围字段；不能由商品编号
或时间推测、重建已经丢失的会话引用。

此旧直接游标接口仅限 admin/Content，排队和网络共用 60 秒调用预算。首读及有限
重连发送前均须至少剩余 10 秒网络预算；首读排队后不足时返回
`503 CONVERSATION_READ_DEFERRED`，其中 `reason=read_budget_insufficient`、
`read_sent=false`、`retryable=true`。这表示本次读取没有发给上游，并非原生成失败；
调用方按既有退避稍后重查同一游标，不新建或重发生成。如果第一次连接已失败，
后续预算不足仍保留原连接错误，不能谎报整次调用未发出。普通调用者的持久任务
查询与恢复流程不受此接口调整影响。

只有原锚点仍在当前分支、没有后续用户消息，且当前消息是已结束的完整 assistant
正文，才返回 `status=succeeded`、`binding_status=bound` 和正文。分析文本、未结束
或空正文返回 `status=running`，不能当作生成成功。账号或会话错配返回 409。

初始文本流异常且已取得会话编号时，服务先查询一次既有结果；未完成则保留原引用供
后续 GET 查询，不重复生成。该自动恢复不用于后续聊天轮次，避免取到前一轮回答。
Workbench 消费者需先部署此 Provider 接口，再启用保存和查询原文案引用的 Worker；
历史记录缺少原引用时仍是结果未知，不能自动重提。
