太棒了！🎉  
欢迎来到 **AI 学习第 4 天**，我们今天正式进入 **现代深度学习的核心技术** —— **Transformer 与 Self-Attention**。

---

# 📅 第4天：理解 Transformer & 自注意力（Self-Attention）

---

## 🎯 今日学习目标

你将理解：

- 👁 什么是 Self-Attention？
- 🧠 Transformer 的结构和工作原理
- 🤖 为什么它能取代 RNN 和 CNN？
- 🔍 实现一个简化版的 Self-Attention 模块（PyTorch）

---

## 📍 背景回顾：RNN 的局限

在 Transformer 出现之前，NLP 任务主要通过 RNN / LSTM 处理序列数据，但它们有很多问题：

| 问题 | 描述 |
|------|------|
| 顺序依赖 | 无法并行处理 |
| 长距离依赖难 | 句子太长时，前后信息难以关联 |
| 训练慢 | 逐步处理，不能 GPU 并行加速 |

---

## 🌟 Transformer 的革命性设计

> 2017 年，论文《Attention is All You Need》提出了 Transformer 架构，完全抛弃了 RNN，使用了 **Self-Attention** 机制。

---

## 🧠 什么是 Self-Attention（自注意力）？

> Self-Attention 是一种机制，它让每个词在句子中都能“关注”其他词，从而捕捉上下文信息。

---

### 举个例子：

句子：**“The cat sat on the mat.”**

- Self-Attention 会让 “cat” 关注 “sat”，
- “mat” 关注 “on the”，
- 每个词都能结合上下文重新理解自己

---

## 🧮 Self-Attention 的计算公式

对于每个词（向量）：

1. 计算三个向量：
   - Query（Q）
   - Key（K）
   - Value（V）

2. 计算注意力权重：

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V
\]

- \( QK^T \)：衡量词与词的相关性
- \( \sqrt{d_k} \)：缩放因子，避免梯度爆炸
- \( \text{softmax} \)：标准化成概率
- 最终加权得到输出

---

## 📦 Transformer 的核心结构

```
输入 → 位置编码 → 多头注意力 → 前馈网络 → 残差连接 + LayerNorm → 输出
```

| 模块 | 作用 |
|------|------|
| 位置编码 | 加入位置信息（因为没有 RNN 的顺序） |
| 多头注意力 | 并行多个 self-attention，捕捉不同角度 |
| 前馈网络 | 每个位置独立做映射（MLP） |
| 残差连接 | 防止梯度消失，加快训练 |
| LayerNorm | 稳定训练过程 |

---

## 👨‍💻 PyTorch 实现一个简化版 Self-Attention 层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        B, T, C = x.shape
        qkv = self.qkv(x)  # -> (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # 分成 Q, K, V

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # 注意力权重
        attn_output = torch.matmul(attn_weights, v)  # (B, T, C)

        return self.out(attn_output)  # 输出映射
```

---

## 📈 Self-Attention 的优势

| 优点 | 说明 |
|------|------|
| 并行计算 | 不需要逐步处理序列，支持全 GPU 并行 |
| 更长依赖捕捉 | 每个词都能看全句子 |
| 架构灵活 | 可扩展为 BERT、GPT、ViT 等模型 |

---

## 🧠 总结口诀：

> **“全句互相关，注意力计算强；位置补信息，残差保梯度。”**

---

## ✅ 今日任务 checklist

| 任务 | 完成 |
|------|------|
| 理解 Self-Attention 原理 ✅ | [ ] |
| 熟悉 Transformer 架构 ✅ | [ ] |
| 实现简化版 Self-Attention ✅ | [ ] |
| 知道它为何比 RNN 更强 ✅ | [ ] |

---

## 🔜 明日预告（Day 5）

> 手把手训练一个 **Transformer 文本分类模型**，体验现代 NLP 的力量！

---

