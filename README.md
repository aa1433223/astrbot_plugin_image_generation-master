# 通用生图插件

AstrBot 通用图像生成插件，支持文生图、图生图、LLM 工具调用、多供应商配置、预设提示词、用户限制和安全审核。

> 当前仓库是 `noram` 本地签名版。`metadata.yaml` 使用唯一插件名 `noram_image_generation`，并清空 `repo`，用于避免 AstrBot 市场按原仓库自动提示更新并覆盖本地修改。页面展示名仍为“通用生图”。

## 快速使用

### 指令

```text
/生图 一只在窗边晒太阳的橘猫
/生图 9:16 4K 一张竖版赛博朋克城市海报
/生图 4k 9;16 手办化 透明亚克力底座，桌面背景
/生图 手办化 4K 9：16 透明亚克力底座，桌面背景
/生图 --ar 16:9 --res 2K 一张宽屏电影感风景
/生图 手办化 透明亚克力底座，桌面背景
/生图模型
/生图模型 2
/预设
/预设 添加 赛博朋克:cyberpunk city, neon lights
/预设 添加 竖版海报 9:16 4K cinematic poster, neon lights
/预设 删除 赛博朋克
```

- `/生图 [宽高比] [分辨率] <提示词或预设名称> [额外提示词]`：启动生图任务。宽高比和分辨率可省略，省略时使用默认配置。
- `/生图模型`：查看当前可用模型，模型格式为 `供应商名称/模型名称`。
- `/生图模型 <序号>`：切换模型，并刷新 LLM 工具 schema。
- `/预设`：管理提示词预设。

`/生图` 支持传入 `9:16`、`16:9`、`4K` 等参数，也支持 `--ar 9:16 --res 4K` 或 `--aspect-ratio=9:16 --resolution=4K`。普通提示词只在开头识别参数，避免误删正文里的 `4K`、`16:9`；如果命中了预设名，参数可以写在预设名前或后。中文冒号 `9：16`、英文/中文分号 `9;16` / `9；16` 和小写 `4k` 会自动规范化。

旧预设格式 `名称:长提示词` 完全兼容，只按第一个冒号拆分名称和内容，正文里的英文冒号、中文冒号、Markdown 标题会保留。新预设也可以写成 `/预设 添加 名称 9:16 4K 提示词`，插件会保存为带宽高比和分辨率的结构化预设。

如果消息、引用消息或 @ 用户里带有图片，且当前适配器支持图生图，插件会自动把图片或头像作为参考图。

参考图会优先读取插件本地缓存；同一来源已有有效缓存时会直接复用。未命中或缓存校验失败时，会重新写入缓存并读回，确认缓存成功后才会启动图生图任务。当前本地签名版的 `metadata.yaml` 插件名是 `noram_image_generation`，所以运行时缓存目录通常是：

```text
AstrBot/data/plugin_data/noram_image_generation/cache
```

插件加载时日志会打印实际 `data_dir` 和 `cache_dir`。如果你在 `data/plugin_data/astrbot_plugin_image_generation/cache` 看不到参考图，请优先按日志里的实际路径查找。

### 触发说明

插件同时支持：

- `@filter.command` 指令触发
- 被动监听兜底（当部分环境 command 触发不稳定时）

因此在不同适配器环境中，`/生图`、`/生图模型` 和 `/预设` 都有更高概率稳定触发。兜底命中后会停止事件继续传播，避免长提示词同时进入普通聊天链路。

### LLM 工具

开启 `enable_llm_tool` 后，LLM 可以调用 `generate_image`：

| 参数 | 说明 |
| --- | --- |
| `prompt` | 必填，生图提示词 |
| `aspect_ratio` | 可选，`自动`、`1:1`、`16:9`、`9:16` 等 |
| `resolution` | 可选，`1K`、`2K`、`4K` |
| `avatar_references` | 可选，`self`、`sender` 或 QQ 号，用头像作为参考图 |

工具参数会按当前模型能力自动裁剪。例如当前适配器不支持图生图时，`avatar_references` 不会暴露给 LLM。

## 适配器支持

| 适配器 | 文生图 | 图生图 | 宽高比 | 分辨率 | 说明 |
| --- | :---: | :---: | :---: | :---: | --- |
| `openai` | 支持 | 支持 | 支持 | 支持 | OpenAI / NewAPI / OpenAI 兼容生图接口 |
| `gemini` | 支持 | 支持 | 支持 | 支持 | Gemini 原生接口 |
| `gemini_openai` | 支持 | 支持 | 按配置 | 按配置 | 通过 OpenAI 兼容 chat/completions 调 Gemini 生图 |
| `grok` | 支持 | 支持 | 支持 | 支持 | xAI Grok 图像接口 |
| `z_image_gitee` | 支持 | 不支持 | 支持 | 支持 | Gitee AI Z-Image |
| `jimeng2api` | 支持 | 支持 | 支持 | 支持 | Jimeng2API 逆向接口，配置后启动和每日会自动领取积分 |

OpenAI 风格的 `data[].b64_json` / `data[].url` 图片回收逻辑已抽到公共能力，`gemini_openai`、`grok`、`z_image_gitee`、`jimeng2api` 的 URL 图片下载会复用适配器代理和统一下载超时。

## OpenAI / NewAPI

### 文生图

OpenAI / NewAPI 文生图走：

```text
POST /v1/images/generations
```

插件会把用户侧的 `aspect_ratio + resolution` 转成合法的 OpenAI `size=WIDTHxHEIGHT`。例如：

| 用户参数 | NewAPI 中转站收到的 `size` |
| --- | --- |
| `16:9 + 4K` | `3840x2160` |
| `9:16 + 4K` | `2160x3840` |
| `1:1 + 2K` | `2048x2048` |
| `自动 + 1K` | `1024x1024` |

非官方 OpenAI `base_url` 默认启用精确尺寸透传，最长边限制为 `3840`，避免中转站报 `The longest edge must be less than or equal to 3840`。

### 图生图

默认图生图路径：

```text
POST /v1/images/edits
multipart/form-data
```

OpenAI/NewAPI 只要请求里有参考图，默认都会走 multipart，把图片作为重复 `image=@photo.png` 文件字段提交，不使用 `image[]`。参考图最多保留前 4 张。这样最容易确认中转站实际收到图片。

参考图会先规范化为 PNG/JPEG；GIF、WebP、HEIC/HEIF 等可解析格式会自动转 JPEG，无法解析的图片会跳过并写日志。参考图最多保留前 4 张。

只要请求里已有参考图，OpenAI/NewAPI 适配器会强制走 multipart edits 图生图路径；如果参考图全部无法解析，会直接报错，不会静默降级为文生图。

> **注意**：`/v1/images/edits` (multipart) 不支持 `wait:false` 异步模式，因此图生图始终是同步请求。文生图的 `newapi_async` 异步模式不影响图生图。

### 结果格式与回退

- `prefer_url_response=false`：保持默认行为。
- `prefer_url_response=true`：优先请求 `response_format=url`，插件再本地下载图片。
- 如果 URL 路线下载失败，或 NewAPI 返回 `download_failed`、`invalid_response`、`invalid_request_error`，插件会自动重试一次 `b64_json`。
- 只回退一次，避免循环和重复计费失控。

### 详细日志

`trace_mode` 是 OpenAI 专用排障开关。开启后会写入：

```text
插件数据目录/logs/openai_trace.log
```

日志包含提交请求、HTTP 状态、完整 API 原始响应、轮询状态、URL 下载耗时、回退原因等。图生图请求默认应看到 `submit_mode=edits_multipart`、`multipart_image_field_count > 0` 和 `multipart_images` 的文件名/MIME/字节数；显式使用 JSON 模式时应看到 `submit_mode=generations_json_references` 且 `json_reference_image_count > 0`。如果 API 响应里 `image_tokens=0`，trace 会记录 `reference_images_ignored`。按要求会完整保存 API body，包括 `b64_json` 图片数据，所以文件可能很大；排查结束后建议关闭 `trace_mode`。

## 配置说明

### 基础配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enable_llm_tool` | `true` | 是否允许 LLM 自动调用生图工具 |

### NewAPI Key 分发接入

如果同时安装了 `astrbot_plugin_newapi_key_distributor`，可以让每个 QQ 用户使用自己的 NewAPI Key 生图，而不是共用本插件 `api_keys`。

Key 分发插件必须开启：

```text
store_plain_keys = true
```

本插件开启：

```text
key_distributor.enabled = true
key_distributor.require_key = true
key_distributor.data_path =
```

`data_path` 留空时会默认读取同级插件数据目录下：

```text
astrbot_plugin_newapi_key_distributor/newapi_key_distributor.json
```

启用后，`/画图`、`/生成图` 和 LLM 生图工具会按当前 QQ 查找 active 且保存了明文 `key_plain` 的 Key，并仅在本次 OpenAI/NewAPI 请求里使用该用户自己的 Key。当前只对 `openai` 适配器生效。

### API 供应商

每个供应商包含：

| 配置项 | 说明 |
| --- | --- |
| `name` | 供应商名称，模型选择格式为 `供应商名称/模型名称` |
| `base_url` | API Base URL，可填中转站地址 |
| `proxy` | 代理地址，支持 `http://127.0.0.1:7890`、`socks5://127.0.0.1:7891` |
| `api_keys` | API Key 列表 |
| `available_models` | `/生图模型` 可切换的模型列表 |
| `capability_options` | 按模型实际情况声明文生图、图生图、宽高比、分辨率能力 |

OpenAI 额外配置：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `model_family` | `auto` | `auto` 会按模型名识别 `gpt-image` 或 `dall-e` |
| `prefer_url_response` | `false` | 优先 URL 回收图片，失败时自动回退一次 `b64_json` |
| `newapi_async` | `true` | 非官方 OpenAI `base_url` 文生图默认用 `wait:false` 异步提交并轮询；图生图始终同步，不受此选项影响 |
| `connect_timeout` | `30` | TCP/TLS 或代理隧道建连超时，避免握手卡满生成总超时 |
| `proxy_fallback_direct` | `true` | 代理建连失败时自动绕过代理直连重试一次 |
| `trace_mode` | `false` | 写入 OpenAI 详细 trace 文件 |

### 生成行为

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | 空 | 当前模型，留空使用第一个可用模型 |
| `timeout` | `180` | 请求或轮询总超时，OpenAI 精确尺寸中转站至少按 180 秒处理 |
| `max_retry_attempts` | `3` | 最大重试次数 |
| `max_concurrent_tasks` | `3` | 最大并发生图任务数 |
| `default_aspect_ratio` | `自动` | 命令生图默认宽高比 |
| `default_resolution` | `1K` | 命令生图默认分辨率 |
| `show_generation_info` | `false` | 成功后显示耗时和图片数量 |
| `show_model_info` | `false` | 成功后显示模型名称 |

### 用户限制与审核

- `umo_blacklist`：会话黑名单，命中后指令和 LLM 工具都会被拦截。
- `blacklist_block_message`：黑名单提示语，留空则不发送文本。
- `rate_limit_seconds`：单个会话两次请求间隔。
- `enable_daily_limit` / `daily_limit_count`：每日生图次数限制。
- `safety_audit.umo_whitelist`：审核白名单。
- `prompt_audit`：生图前提示词审核，支持关键词和 AI 审核。
- `image_audit`：生图后图片审核，发送前调用对话模型判断图片是否可发。

## 预设

简单格式：

```text
名称:提示词
```

高级 JSON 格式：

```text
表情包:{"prompt":"为图中角色生成 Q 版表情包","aspect_ratio":"16:9","resolution":"2K"}
```

使用：

```text
/生图 表情包 增加一些开心和惊讶表情
```

## 排障建议

- 中转站没有收到参考图：开启 `trace_mode`，图生图默认必须看到 `submit_mode=edits_multipart`、`multipart_image_field_count > 0`，以及 `multipart_images` 里每张图的文件名/MIME/字节数。
- 中转站 60 秒左右 504，但后台实际出图：文生图可确认 `openai.newapi_async=true`；图生图始终走 multipart 同步模式，受 Cloudflare 等 CDN 超时限制（通常 120s），如果中转站处理时间超过 CDN 超时会返回 524 错误。
- 报 `invalid_reference_image` / 参考图解析失败：检查上传图片是否超过 `max_image_size_mb`、是否能被 PIL 解析；OpenAI/NewAPI 路径会自动转为 PNG/JPEG，multipart 字段应为重复 `image` 而不是 `image[]`。
- 参考图看起来像压缩图：插件会优先使用 `path`、本地 `file`、NapCat/aiocqhttp `get_image(file=...)` 返回的本地文件；如果只能使用消息段 URL，日志会提示 `component_url_fallback`，这取决于平台是否提供原图。
- 找不到参考图缓存或怀疑用了旧图：看插件加载日志里的实际 `cache_dir`。每张参考图会按来源生成稳定缓存键，命中有效的 `ref_*.png/.jpg/.gif/.webp/.heic/.heif` 时直接读取；未命中或缓存校验失败时才写入 `ref_*.tmp`，校验成功后原子替换为最终缓存文件。
- 报 `Invalid size '16:9'`：说明没有走 `resolution/aspect_ratio -> WIDTHxHEIGHT` 映射，检查当前模型是否使用 `openai` 适配器。
- 图片生成成功但回收慢：尝试开启 `prefer_url_response`，并配置 `proxy`。
- 报 `Cannot connect to host ... ssl` / `SSL handshake`：说明请求还没到中转站。配置了 `proxy` 时会自动直连重试一次；trace 里应看到 `proxy_connect_failed` 和 `proxy_fallback_direct_success`。如果部署环境必须强制走代理，可以关闭 `proxy_fallback_direct`。
- 需要定位完整链路：开启 `trace_mode`，查看 `logs/openai_trace.log`。

## 本地验证

```text
python -m py_compile astrbot_plugin_image_generation\core\base_adapter.py astrbot_plugin_image_generation\core\types.py astrbot_plugin_image_generation\adapter\openai_adapter.py
pytest -q tests\test_openai_adapter_size.py
python -m json.tool astrbot_plugin_image_generation\_conf_schema.json
```
