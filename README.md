# Model Sentinel

用少量短探针，检测模型接口相对健康基线的能力回退。Python 3.10+，零第三方运行时依赖，兼容 Chat Completions 风格接口。判分由确定性代码完成。

## 建立基线并检查

```powershell
$env:OPENAI_API_KEY = "你的密钥"
$env:MODEL_SENTINEL_MODEL = "你的模型名"

# 每题三次全部通过、且无请求错误，才成为健康基线探针
python -m model_sentinel run --repeats 3 --output baseline.json

# 默认 standard：分层抽样、失败确认、SPRT 提前停止
python -m model_sentinel check --baseline baseline.json --output check.json
```

本地服务可用 `--base-url http://127.0.0.1:8000/v1 --model local-model`，无需设置密钥。显式指定 `--api-key-env MY_KEY` 时只读取该变量；未指定时依次查找 `MODEL_SENTINEL_API_KEY`、`OPENAI_API_KEY`。

**0.2 版本使用报告 schema v2，旧版报告必须重新采集。** 旧报告未记录完整调用条件，无法可靠补齐；仓库保留的 Codex CLI 冒烟记录仅供历史参考，不能作为 API 检测基线。

0.2.1 保持 schema v2 与评分规则兼容，已有 0.2.0 基线可继续使用。命令在请求前检查输出路径是否可写，并拒绝用输出覆盖本次输入的题包、基线或候选报告；`calibrate` 同样保护标签清单及其引用文件。

## 三种检查模式

| 模式 | 命令 | 结果含义 |
|---|---|---|
| quick | `check --mode quick --output smoke.json` | 最多 12 题的绝对冒烟测试，不需要基线；输出 `smoke_pass / smoke_fail / smoke_error` |
| standard | `check --baseline baseline.json --output check.json` | 自适应检测；默认最多 32 个有效观察，正常约 12 题停止 |
| deep | `check --mode deep --baseline baseline.json --output deep.json` | 全题包、默认每题两次采样，完整配对比较；报告包含候选原始结果 |

`quick` 的通过不代表相对基线稳定，冒烟报告也不能作为基线。`deep --repeats 3 --workers 4` 可增加采样并启用有界并发；第三次及后续采样进入诊断和基线稳定性统计，失败确认事件仍使用前两次，保持与在线检测一致。`standard` 保持顺序执行。

## 比较条件与判定规则

报告记录题包摘要、评分器版本、接口类型与端点、模型、system prompt 摘要、温度、模型 seed、推理强度及各题有效输出上限。基线与检测配置默认必须一致。需要有意比较其他模型或参数时，使用 `--allow-config-mismatch`，差异会保存在报告中。

推理模型可按供应商支持情况设置：

```text
--token-param max_completion_tokens --max-tokens 256 --omit-temperature
--reasoning-effort low
```

基线与后续检测必须使用一致配置。程序记录截断和拒绝等状态，默认将它们作为未解决的观察，不直接计入能力回退；不会自动提高某一次检测的输出预算。

默认失败确认口径在在线、离线两种路径中一致：首次通过算无回退；首次失败且第二次也失败算确认回退；第二次通过记为 flaky；请求错误或缺少第二次结果则不构成确认。`--no-confirm` 改为单次观察，需要针对该策略重新校准阈值。

离线比较使用**全部有效配对**，不会忽略后半段已知失败：

```powershell
python -m model_sentinel run --repeats 2 --output candidate.json
python -m model_sentinel compare baseline.json candidate.json --output comparison.json

# 显式回放顺序检测，同时保留全量摘要
python -m model_sentinel compare baseline.json candidate.json --replay
```

默认离线规则是精确二项尾检验；`--replay` 使用 SPRT。两者共享“回退观察”的定义，但停止规则和最终决策可能不同。离线比较不受 `--max-samples` 截断，该参数仅约束顺序检测。详细规则与假设见 [设计文档](docs/DESIGN.md)。

## 覆盖、错误和预算

`standard` 默认按类别分层随机排列，每次自动生成并保存抽样 seed。使用 `--sample-seed 42` 可以重放题目顺序；它独立于传给模型的 `--seed`。`--sampling ordered` 使用题包顺序。默认每个类别至少需要一个有效观察才允许报告稳定，可用 `--min-category-samples` 调整。

统计量达到稳定边界时，如果类别覆盖或错误比例仍不满足条件，检测会在剩余题目和预算内继续采样，直到满足条件或达到停止限制。补采结果继续参与统计，可能改变最终结论；明确的回退证据仍会及时停止。

报告分别记录能力判定、可用性、有效覆盖、flaky、类别汇总和未解决观察。默认未解决比例超过 `--max-error-rate 0.2` 时，不允许给出稳定结论。对重要业务应定期运行 `deep`，并用有标签的健康/退化历史报告校准参数，避免把短题包的结论扩展到全部业务能力。

```powershell
python -m model_sentinel check --baseline baseline.json --output check.json `
  --max-requests 80 --deadline 300 --timeout 30 --retries 2
```

`--max-requests` 包含 HTTP 重试及失败确认；`--deadline` 为整个客户端生命周期的秒数预算，`--timeout` 限制单次网络等待。预算耗尽或遇到不可恢复的认证/配置错误时停止新请求并保存部分结果。全量并发运行可能仍有已经发出的请求，但共享请求预算不会被突破。

运行期限在请求、退避、分块读取和返回前检查；同步 DNS 与响应头读取仍受底层网络超时约束，不能保证操作系统级的硬终止。服务端的 `Retry-After` 同样约束后续探针和并发线程，包括不再重试的错误响应。如果要求等待超过 60 秒，当前运行会保存中止结果，由后续调度重试。TLS 证书验证失败会立即中止，避免重复发送无效请求。

报告保存包含重试等待的总耗时、HTTP 尝试信息和嵌套 token 明细。即使响应字段异常或解析后超过运行期限，已收到的合法 usage 仍计入总量；无法取得有效 usage 的请求标记为用量未知，已知 token 总数不代表这些请求没有成本。HTTP 重试支持 `Retry-After` 和随机退避。客户端禁止自动重定向，并在保存错误信息前脱敏。

| 退出码 | 含义 |
|---:|---|
| 0 | 检测稳定、冒烟通过，或全量报告采集成功 |
| 1 | 参数/文件/执行错误，包括请求失败和预算中止；已获得的能力证据仍保存在报告中 |
| 2 | 回退或冒烟失败 |
| 3 | 证据不足 |

`run` 用于采集基线或候选报告，因此答错本身不导致非零退出码；应使用 `check` 或 `compare` 进行门禁判定。

## 用历史数据校准

准备标签清单 `labels.json`，路径相对该清单所在目录：

```json
{"cases":[
  {"name":"healthy","baseline":"baseline.json","candidate":"healthy.json","expected":"stable"},
  {"name":"degraded","baseline":"baseline.json","candidate":"degraded.json","expected":"regressed"}
]}
```

```powershell
python -m model_sentinel calibrate labels.json --output calibration.json
```

清单可选顶层 `pack` 指定自定义题包。命令使用已保存报告，输出混淆矩阵、误报率、漏报率和证据不足率；支持相同的统计参数、确认策略与 `--replay`。它不会调用模型或自动更改阈值。结果仅描述当前标签集；调参后应使用独立留出的历史样本验证。

同一次校准中，共享的报告文件只读取、解析一次；再次执行命令时重新读取，保证文件更新生效。

## 自定义题包

内置 32 题覆盖算术、逻辑、代码、指令遵循和结构化输出。可复制 `model_sentinel/packs/core_v1.json`，加入稳定、可独立核验的私有业务题。每次更改题包或评分规则后重新建立基线。

```json
{"type":"exact","answers":["PAES"],"case_sensitive":true,"normalize_whitespace":false,"normalize_unicode":false}
```

`exact` 默认忽略大小写、归一化空白和 Unicode 宽度。对代码、标识符或格式敏感题，请显式关闭相应宽松处理。

```json
{"type":"json","value":{"x":7,"ok":true}}
```

JSON 比较区分布尔与数字、字符串与数字，忽略对象键顺序，保留数组顺序；`1` 与 `1.0` 作为相同 JSON 数字。重复键、非有限数值及缺少标准答案均被拒绝；显式的 `"value": null` 合法。

```powershell
python -m model_sentinel validate-pack my_pack.json
python -m unittest discover -s tests -v
```

CI 覆盖 Windows/Linux 和 Python 3.10/3.12，并检查 wheel 安装后的命令行入口及内置题包。开发进度和剩余扩展方向见 [opti.md](opti.md)。
