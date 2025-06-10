
# 📅 Day3：激活函数、损失函数与优化器（构建能“学习”的神经网络）

---

## 🎯 今日学习目标

> 理解神经网络中的三个关键角色：
>
> - **激活函数**：让网络更聪明（非线性）
> - **损失函数**：告诉模型“你错了多少”
> - **优化器**：教模型“如何变得更好”

你将实现一个可学习的神经网络小模型，并用 PyTorch 训练它完成分类任务！

---

## 🧠 一、激活函数（Activation Function）

### 🔹 是什么？
> 激活函数是每个神经元的“开关”，决定是否激活该神经元。

### 🔹 为什么需要？
> 没有激活函数，神经网络就只是一个线性函数，无法学习复杂的非线性关系。

### 🔹 常见激活函数：

| 名称 | 公式 | 特点 |
|------|------|------|
| **ReLU** | \( f(x) = \max(0, x) \) | 快速、简单，默认首选 |
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | 输出在 (0,1)，适合二分类 |
| **Tanh** | \( f(x) = \tanh(x) \) | 输出在 (-1,1)，对称 |

---

## 🎯 小实验：激活函数可视化

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

plt.plot(x, relu, label="ReLU")
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")
plt.legend()
plt.title("常见激活函数")
plt.grid(True)
plt.show()
```

---

## 💥 二、损失函数（Loss Function）

### 🔹 是什么？
> 衡量模型输出结果与真实标签之间差异的函数，越小越好。

### 🔹 常见类型：

| 类型 | 使用场景 | PyTorch 中的实现 |
|------|----------|-------------------|
| **MSELoss** | 回归问题 | `nn.MSELoss()` |
| **CrossEntropyLoss** | 多分类 | `nn.CrossEntropyLoss()` |
| **BCELoss** | 二分类 | `nn.BCELoss()` |

### 🧠 举例：
- 预测猫是 90%，实际是猫（1），损失就很小
- 预测是狗（0.1），但实际是猫，损失就大

---

## ⚙️ 三、优化器（Optimizer）

### 🔹 是什么？
> 优化器通过“调整权重”，让损失函数越来越小。

### 🔹 常见优化器：

| 名称 | 特点 | PyTorch 中的写法 |
|------|------|------------------|
| **SGD** | 最基本的梯度下降 | `torch.optim.SGD()` |
| **Adam** | 自带动量、自适应学习率 | `torch.optim.Adam()` |
| **RMSprop** | 适用于 RNN | `torch.optim.RMSprop()` |

---

## 👨‍💻 四、实战任务：训练一个简单神经网络（二分类）

我们使用 **PyTorch** 实现一个分类模型：识别一个点是否在圆内。

### ✅ 数据生成 + 模型训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据：判断点是否在单位圆内
torch.manual_seed(42)
x = torch.rand(500, 2) * 2 - 1  # [-1, 1]
y = (x[:, 0]**2 + x[:, 1]**2 < 0.5).float().unsqueeze(1)  # 0 或 1

# 定义模型
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# 损失函数 & 优化器
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
losses = []
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    losses.append(loss.item())

# 可视化损失曲线
plt.plot(losses)
plt.title("训练损失下降曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

---

## ✅ 今日任务清单

| 任务 | 状态 |
|------|------|
| 理解激活函数的作用 ✅ | [ ] |
| 用代码可视化激活函数 ✅ | [ ] |
| 理解损失函数的意义 ✅ | [ ] |
| 学会使用 Adam 优化器 ✅ | [ ] |
| 使用 PyTorch 训练一个神经网络 ✅ | [ ] |
| 写一段总结（每个模块的作用） ✅ | [ ] |

---

## 🧠 总结口诀：

> **激活让网络更聪明，损失告诉它哪里错，优化器教它怎么变好。**

---

## 🎯 明日预告（Day 4）

> 了解 Transformer 架构的关键模块：**Self-Attention**，并理解它如何改变 NLP 的未来！

