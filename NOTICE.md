# 上游复用说明

本材料为1到2工程集成探索，保存了研究原型及原始对照数据。

- `workspace_policy.py`中的范围估算复用FlashInfer PR #5236的分配器算术，提交`fdf3c5142b27dc34b8c2cb1aa8fe8910add79486`。来源、原始摘录、许可归属和摘要见`policy-evidence.json`及原文件头。
- `vendor/pr5236/scheduler.cuh`与`utils.cuh`保留原始文件头。上游FlashInfer采用Apache-2.0许可，版权归原文件列出的贡献者。
- 具体计划查询使用FlashInfer既有`workspace_size()`；数值基线复用PR #4401的修复。vLLM执行、显存分析、Graph和原生请求计时直接复用既有实现。
- 新增工作集中于启动选择接入、支持范围与回退、缓存容量和实际生成对照、结果复算及可复现交付。原型编写和分析使用AI协助。

上游链接：

- https://github.com/flashinfer-ai/flashinfer/pull/5236
- https://github.com/flashinfer-ai/flashinfer/pull/3741
- https://github.com/flashinfer-ai/flashinfer/pull/4401
- https://github.com/vllm-project/vllm/pull/28269
- https://github.com/vllm-project/vllm/pull/46883
- https://github.com/vllm-project/vllm/pull/40383
