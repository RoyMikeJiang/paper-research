---
tags:
  - paper
  - simplified
aliases:
  - APG 简明版
---

# APG —— 实例级动态参数生成的 CTR 预测

> 原始论文笔记：[APG](APG.md)

## 一句话总结

传统深度 CTR 模型对所有输入共享一套静态权重。APG 为每个输入样本动态生成专属参数，通过低秩分解 + 参数共享的设计，最终实现比静态模型更快（-38.7%）、更省内存（-96.6%）、更准（+0.52% AUC）。已在阿里巴巴直通车线上部署，CTR +3%。

## 问题背景

CTR（点击率）预测是推荐/广告系统的核心——预测用户是否会点击某个广告或商品。主流模型（DeepFM、DCN、PNN 等）都用**一套静态参数 $W$** 处理所有输入：

$$y_i = f_W(x_i)$$

问题在于：冷启动用户（2 次交互）和活跃用户（2000 次交互）的特征分布差异巨大，一套参数只能在这些分布之间求妥协。

粗粒度方案（STAR 按域分参数、MMoE 按任务分专家）无法扩展到实例级——STAR 在 item 级粒度下需要 13,754 MB 内存。而朴素的实例级参数生成（为每个样本生成完整 $W_i$）导致训练时间 ×111、内存 ×31。

## 核心方案

APG 将模型从静态参数改造为动态参数：

$$y_i = f_W(x_i) \quad \Rightarrow \quad y_i = f_{G(z_i)}(x_i)$$

其中 $G$ 是参数生成网络，$z_i$ 是条件信号（可以是用户特征、上一层输出等）。

![apg/images/fig_framework.png](apg/images/fig_framework.png)
*APG 整体框架：条件信号 $z_i$ 经生成网络产生实例特定参数 $S_i$，与共享参数 $U, V$ 组合为动态权重*

### 四步优化

![apg/images/fig_optimization_steps.png](apg/images/fig_optimization_steps.png)
*四步渐进式优化：从静态参数 → 朴素实例参数 → 低秩分解 → 参数共享 → 过参数化*

**Step 1 — 低秩分解**：不生成完整 $N \times M$ 矩阵，而是分解为三个小矩阵：

$$W_i = U_i \cdot S_i \cdot V_i, \quad U_i \in \mathbb{R}^{N \times K},\ S_i \in \mathbb{R}^{K \times K},\ V_i \in \mathbb{R}^{K \times M},\ K \ll \min(N, M)$$

**Step 2 — 分解前馈**：利用矩阵乘法结合律，右结合逐步计算，避免重建 $N \times M$ 矩阵：

$$y_i = \sigma(U_i \cdot (S_i \cdot (V_i \cdot x_i)))$$

数学上是恒等变换，只节省计算不影响结果。

![apg/images/fig_decomposed_feedforward.png](apg/images/fig_decomposed_feedforward.png)
*分解前馈：左侧先重建完整矩阵再前馈（慢），右侧逐步矩阵乘法（快），结果完全一致*

**Step 3 — 参数共享（最关键）**：
- $U, V$ 设为全局共享的静态参数（所有样本共用）
- 只有 $S_i$（$K \times K$ 小矩阵）由生成网络按样本动态产生

$$S_i = \text{reshape}(\text{MLP}(z_i)), \quad y_i = \sigma(U \cdot (S_i \cdot (V \cdot x_i)))$$

这一步的反直觉发现：准确率从 +0.27%（朴素方案）提升到 +0.40%。原因是共享参数 $U, V$ 能从全量数据中学到稳定的通用表示，$S_i$ 只需要建模实例间的残差差异——分工比"全部独立生成"更高效。

**Step 4 — 过参数化**：训练时将共享参数扩展为两个矩阵的乘积：

$$U = U^l \cdot U^r, \quad U^l \in \mathbb{R}^{N \times P},\ U^r \in \mathbb{R}^{P \times K},\ P \gg K$$

推理前预乘回 $U = U^l U^r$。训练时有更大的优化空间，推理时零额外开销。

![apg/images/fig_train_vs_inference.png](apg/images/fig_train_vs_inference.png)
*过参数化：训练时 $U, V$ 分解为两个矩阵提供更大优化空间，推理时预乘回单一矩阵*

### 效率对比

以层维度 $N=1024, M=512$ 为例：
- 静态模型内存：$N \times M = 524,288$ 个参数
- APG 内存（$K=4, D=32$）：$K^2 D + NK + MK = 6,656$ 个参数

约 **80 倍**的内存节省。最终 APG 的训练速度比静态模型快 38.7%，内存少 96.6%。

## 实验结果

- **通用性**：在 7 种 CTR 模型 × 3 个数据集上全部有效，平均 AUC 提升 0.24%-0.91%
- **工业验证**：阿里巴巴直通车线上 A/B 测试 CTR +3%，RPM +1%，延迟 14.8ms
- **vs 粗粒度方案**：APG 以 0.24 MB 内存达到 79.58% AUC，STAR 需要 3,784 MB 才达到 79.28%（15,000 倍内存差距）
- **冷启动用户获益最大**：离线 AUC 提升 +0.58%，线上 CTR 提升超过 4%

![apg/images/fig_pca.png](apg/images/fig_pca.png)
*生成参数 $S_i$ 的 PCA 可视化：语义相似的商品类目在参数空间中自然聚类，无需显式监督*

## 局限

- 超参数（条件策略、秩 $K$、过参数化因子 $P$）需要手动调整
- 公开小数据集上提升幅度较小（0.24%），工业规模才显著
- 未讨论条件信号信息量不足时的退化行为
