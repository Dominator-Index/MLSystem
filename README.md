# MLSystem

机器学习系统、LLM推理与工程实践的阅读资料和代码入口。

## 本仓库资料

- **[Transformers：Special Tokens与Chat Template](transformers/special_tokens/special_tokens.zh-CN.md)** · [English](transformers/special_tokens/special_tokens.en.md)
- **[KV Cache：AMS区域化缓存压缩](KV-Cache/NeurIPS2026-AMS/README.md)** · [PDF](KV-Cache/NeurIPS2026-AMS/paper.pdf)

## 相关代码库与学习资源

| 资源 | 方向 | 本机checkout |
|---|---|---|
| [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) | ML Systems教程 | `/usr/data/Awesome-ML-SYS-Tutorial` |
| [awesome-production-machine-learning](https://github.com/EthicalML/awesome-production-machine-learning) | 生产机器学习 | `/usr/data/awesome-production-machine-learning` |
| [mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles) | GPU编程与Mojo | `/usr/data/mojo-gpu-puzzles` |
| [agentic-api](https://github.com/vllm-project/agentic-api) | Agentic API与推理服务 | `/usr/data/agentic-api` |
| [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | LLM推理资料 | `/usr/data/Awesome-LLM-Inference` |
| [AdaptiveMassSegment](https://github.com/EIT-NLP/AdaptiveMassSegment) | KV Cache压缩 | `/usr/data/AdaptiveMassSegment` |

## 本地布局

仓库实际位于`/usr/data/MLSystem`；工作区`awesome-exploration-rl/ML-Systems`软链接指向整个仓库。

第三方代码库均保留在`/usr/data`，通过本机软链接访问，不嵌套提交到本仓库。公开入口使用上表的GitHub链接；本机路径映射见[local-links.json](local-links.json)。

2026-10-15迁移保留了已有中英文special tokens笔记及AMS原始PDF。原`ML-Systems/MLSystem`仅是指向本仓库的冗余入口，已在合并时去除，避免仓库内部指向自身。迁移与文件校验记录见[migration-20261015.json](migration-20261015.json)。
