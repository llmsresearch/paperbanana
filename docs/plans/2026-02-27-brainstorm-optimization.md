# PaperBanana 优化分析报告

**日期:** 2026-02-27
**基于:** google-research/papervizagent + dwzhu-pku/PaperBanana 对比分析

## 四维对比评估

### 1. 生成质量

| 差距点 | 官方版 | 我们 | 影响 |
|--------|--------|------|------|
| Critic JSON 结构化 | 严格 JSON + json-repair 容错 | 频繁 parse 失败 | **高** |
| 并行候选生成 | 20 张候选图选最佳 | 单图 | **中** |
| Style Guide | NeurIPS 2025 task-specific .md | 有基础 guidelines | 低 |
| Code-based Plot | matplotlib 代码执行 | Image Gen API | 不采纳 |

### 2. 功能缺失（按优先级）

| 功能 | 优先级 | 决策 |
|------|--------|------|
| Critic JSON 修复 | **P0** | 实现 |
| 并行候选图生成 | **P1** | 实现 |
| 增量 checkpoint | P1 | 待定 |
| Streamlit Demo | P2 | 待定 |

### 3. 我们的架构优势

- Pydantic v2 类型安全（vs raw dict）
- ABC Provider 抽象 + 可扩展
- **Auto VLM 选择**（独创）
- dual-model 架构（Visualizer ≠ Polish）
- structlog / pip 可安装 / MCP / 34+ 测试

### 4. 性能

- Auto VLM 已优化（Flash/Pro 智能切换）
- 批处理缺增量 checkpoint

## 待实现

### Opt-1: Critic JSON 修复

**问题：** Critic agent 返回的 JSON 经常出现 "Unterminated string" 错误，fallback 到 "publication-ready"，导致迭代提前终止。

**方案：**
- 引入 `json-repair` 库（官方版也用）
- 多层 fallback：原始解析 → json-repair → 正则提取 → 默认值
- 考虑使用 Gemini 的 `response_mime_type="application/json"` 强制 JSON 输出

### Opt-2: 并行候选图生成

**问题：** 每轮只生成 1 张图，质量受运气影响大。

**方案：**
- Visualizer 同时生成 N 张候选图（N 可配置，默认 3）
- Critic 评估所有候选图，选出最佳
- 通过 `asyncio.gather()` 并行调用 Image Gen API
