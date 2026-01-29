---
trigger: always_on
---

## 1. Core Principles

1.  **NO YAPPING**
2.  **KISS (Keep It Simple, Stupid)**: 除非必要，否则不引入新的抽象层。
3.  **Workflow**: Analyze (Context) -> Plan (Sequential) -> Execute (Tools/Edit) -> Verify.

## 2. MCP Orchestration

基于你现有的 `mcp_config.json`，严格遵守以下调用逻辑，**禁止滥用工具导致上下文爆炸**。

### 第一梯队：核心思考与上下文

*   **ACE / Context Management** (`acemcp` / `ace-tool`)
    *   **触发**: 需要理解庞大代码库或寻找特定定义时。
    *   **规则**: **优先使用 ACE 索引**而不是让 Windsurf暴力读取大量文件。
    *   **命令**: 使用 `ace-tool` 进行上下文压缩和检索。
*   **Memory** (`memory`)
    *   **触发**: 用户提到偏好、项目特殊约定、长期任务状态。
    *   **规则**: 每次会话开始时读取，结束时存储关键决策。不要把 `memory` 当作临时剪贴板。

### 第二梯队：外部信息获取

*   **Context7** (`context7`)
    *   **触发**: 查询特定的库文档、API 规范（官方源）。
    *   **规则**: 精确查询。不要 dump 整个文档，只获取相关函数的用法。
*   **Code Reasoning** (`code-reasoning`)
    *   **触发**: 分析复杂的静态代码结构、寻找引用链。
    *   **规则**: 在修改核心架构前运行，确保不破坏向后兼容性。
*   **Fetch** (`fetch`)
    *   **触发**: 获取具体的 URL 内容（如 GitHub Raw 文件、简单文档）。
    *   **规则**: 配合 `context7` 使用，如果 `context7` 失败则回退到 `fetch`。
*   **Web Search** (`exa`/`ddg-search`)
    *   **触发**: 寻找最新报错解决方案、库的版本更新。
    *   **规则**: **不要**搜索与当前任务无关的内容，可以使用**模糊**关键词搜索提高debug效率。

### 第三梯队：重型武器

*   **Firecrawl** (`firecrawl-mcp`)
    *   **触发**: 需要爬取整个网页并转换为 Markdown（如教程、博客）。
    *   **警报**: **极其消耗 Token**。
    *   **规则**: 禁止爬取导航栏和页脚。
*   **Arxiv** (`arxiv-mcp-server`/`paper-search-mcp`)
    *   **触发**: 涉及算法研究、论文复现时使用。
    *   **规则**: 只读取 Abstract 和 Conclusion，除非用户明确要求读取全文。

## 3. Workflow Protocol

### Phase 1: Context Gathering
*   不要盲目 `read_file`。先问自己："ACE 索引里有没有？"
*   使用 `memory` 检查是否有历史约定。
*   **输出**: 仅陈述 "Context acquired" 或列出缺失信息。

### Phase 2: Planning
*   调用 `sequential-thinking`。
*   生成步骤列表。
*   **Check**: 这一步是否需要外部联网？(Web Search)。如果需要，立即获取，不要等到写代码时再中断。

### Phase 3: Execution
*   **Edit**: 直接应用变更。对于长文件，使用搜索/替换模式，不要重写整个文件。