# Paper Research

AI 论文阅读笔记，聚焦 **Parameter Generation**（参数生成）方向——用网络生成网络的权重，替代传统的梯度训练范式。

## Papers

| 论文 | 会议 | 年份 | 关键词 | 笔记 |
|------|------|------|--------|------|
| [LoRA-Gen: Specializing LLM via Online LoRA Generation](https://arxiv.org/abs/2506.11638) | arXiv | 2025 | 云端 LoRA 生成、MoE 路由、上下文压缩 | [阅读笔记](LoRA-Gen.md) |
| [IGPG: Instruction-Guided Autoregressive Neural Network Parameter Generation](https://arxiv.org/abs/2504.02012) | arXiv | 2025 | VQ-VAE 权重 tokenization、自回归生成、跨架构泛化 | [阅读笔记](IGPG.md) |
| [APG: Adaptive Parameter Generation Network for CTR Prediction](https://arxiv.org/abs/2203.16218) | NeurIPS | 2022 | 实例级动态参数、低秩分解、工业部署 | [阅读笔记](APG.md) |

## Structure

```
├── LoRA-Gen.md                # LoRA-Gen 阅读笔记
├── IGPG.md                    # IGPG 阅读笔记
├── APG.md                     # APG 阅读笔记
├── lora-gen/
│   ├── paper.pdf              # 原始论文
│   └── images/                # 提取的图表
├── igpg/
│   ├── paper.pdf
│   └── images/
└── apg/
    ├── paper.pdf
    └── images/
```
