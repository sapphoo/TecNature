import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

# plt.plot(x, relu, label="ReLU")
# plt.plot(x, sigmoid, label="Sigmoid")
# plt.plot(x, tanh, label="Tanh")
# plt.legend()
# plt.title("常见激活函数")
# plt.grid(True)
# plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # Step 1️⃣ 生成数据：点是否在单位圆内（中心为0,0，半径 sqrt(0.5)）
# torch.manual_seed(42)
# x = torch.rand(500, 2) * 2 - 1  # 生成 [-1, 1] 范围内的二维点
# y = (x[:, 0]**2 + x[:, 1]**2 < 0.5).float().unsqueeze(1)  # label: 在圆内为1，外为0

# # Step 2️⃣ 定义神经网络模型（MLP）
# model = nn.Sequential(
#     nn.Linear(2, 1),   # 输入层 → 隐藏层（8个神经元）
#     nn.ReLU(),         # 激活函数
#     nn.Linear(1, 1),   # 隐藏层 → 输出层
#     nn.Sigmoid()       # 输出概率
# )

# # Step 3️⃣ 定义损失函数（用于二分类）
# loss_fn = nn.BCELoss()  # Binary Cross Entropy

# # Step 4️⃣ 定义优化器
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Step 5️⃣ 训练模型
# losses = []
# for epoch in range(1000):
#     y_pred = model(x)                 # 前向传播
#     loss = loss_fn(y_pred, y)         # 计算损失
#     optimizer.zero_grad()             # 清空旧梯度
#     loss.backward()                   # 反向传播计算梯度
#     optimizer.step()                  # 更新参数

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
#     losses.append(loss.item())

# # Step 6️⃣ 可视化训练过程
# plt.plot(losses)
# plt.title("训练损失下降曲线")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子，保证可复现
torch.manual_seed(42)

# 1️⃣ 生成数据：判断二维点是否在圆内（半径 √0.5）
x = torch.rand(500, 2) * 2 - 1  # 生成 [-1, 1] 范围内的点
y = (x[:, 0]**2 + x[:, 1]**2 < 0.5).float().unsqueeze(1)  # 在圆内为1，否则为0

# 2️⃣ 拆分训练集/验证集（使用 scikit-learn）
x_train, x_val, y_train, y_val = train_test_split(
    x.numpy(), y.numpy(), test_size=0.2, random_state=42
)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 3️⃣ 定义模型（简单的两层 MLP）
model = nn.Sequential(
    nn.Linear(2, 10000),     # 输入层 → 隐藏层（可改为 1000 看效果）
    nn.ReLU(),
    nn.Linear(10000, 1),     # 隐藏层 → 输出层
    nn.Sigmoid()          # 输出概率
)

# 4️⃣ 定义损失函数和优化器
loss_fn = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5️⃣ 训练模型，并记录训练/验证损失
train_losses = []
val_losses = []

for epoch in range(500):
    model.train()
    y_pred = model(x_train)
    train_loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # 验证模式
    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val)
        val_loss = loss_fn(y_val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# 6️⃣ 可视化训练损失 vs 验证损失
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练 vs 验证损失曲线")
plt.legend()
plt.grid(True)
plt.show()