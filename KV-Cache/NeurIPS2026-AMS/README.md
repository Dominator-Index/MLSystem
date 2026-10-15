# AMS：Adaptive Mass-Segmented KV Compression

**完整题名：** Adaptive Mass-Segmented KV Compression for Long-Form Reasoning。

**认知：** 只对全局token重要性做Top-k，可能把一整段有用的推理上下文清空；压缩预算还需要保留推理区域的结构。

**方法：** 根据attention mass的空间分布自适应分段，为区域分配缓存配额，并用EMA平滑边界；它是可叠加在既有token scorer上的区域分配机制。

- [本地PDF](paper.pdf)：arXiv:2605.23200v2，2026-09-29。
- [arXiv页面](https://arxiv.org/abs/2605.23200v2)
- [官方代码](https://github.com/EIT-NLP/AdaptiveMassSegment)
- 本机代码：`AdaptiveMassSegment`软链接指向`/usr/data/AdaptiveMassSegment`。

本次只迁移原有资料，并按PDF首页补充方法入口，不代表完成全文精读或实验复现。目录沿用原有NeurIPS2026命名，本次未独立核实会议录用状态。

[返回仓库](../../README.md)
