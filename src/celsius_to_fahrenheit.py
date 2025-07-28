import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import matplotlib
from datetime import datetime



# --- 1. 环境与数据准备 ---

plt.style.use('seaborn-v0_8-whitegrid')
# 解决坐标轴负号乱码问题
matplotlib.rcParams['axes.unicode_minus'] = False

# t_c 是摄氏度 (Celsius), 作为模型输入 (x)
# t_u 是华氏度 (Fahrenheit), 作为期望输出 (y)
t_c_list = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0,
            -9.0, -5.5, 1.0, 4.5, 9.0, 12.5, 17.0, 19.5, 24.0, 27.0, 30.0]
t_u_list = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
            14.1, 18.3, 33.5, 38.6, 46.5, 53.6, 61.3, 64.3, 73.1, 80.0, 84.4]

t_c = torch.tensor(t_c_list)
t_u = torch.tensor(t_u_list)

# --- 2. 数据集分割 (训练集与验证集) ---

n_samples = t_c.shape[0]
n_val = int(0.2 * n_samples)  # 20%作为验证集

# 创建随机索引并分割
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

# 创建训练集和验证集张量
train_t_c = t_c[train_indices]
val_t_c = t_c[val_indices]
train_t_u = t_u[train_indices]
val_t_u = t_u[val_indices]

# 对输入数据进行归一化 (如文档中所述，这是一个很重要的步骤)
train_t_cn = 0.1 * train_t_c
val_t_cn = 0.1 * val_t_c


# --- 3. 定义模型、损失函数和训练过程 ---

# 定义线性模型
def model(t_c_normalized, w, b):
    """
    模型函数：根据归一化后的摄氏度预测华氏度
    公式: F = w * (C * 0.1) + b
    """
    return w * t_c_normalized + b


# 定义损失函数 (均方误差)
def loss_fn(t_p, t_u):
    """
    计算预测值 (t_p) 和真实值 (t_u) 之间的均方误差
    """
    squared_diffs = (t_p - t_u) ** 2
    return squared_diffs.mean()


# 定义训练循环
def training_loop(n_epochs, optimizer, params, train_t_c, val_t_c, train_t_u, val_t_u):
    """
    模型训练的核心循环
    """
    train_losses = []
    val_losses = []

    print("开始训练...")
    for epoch in range(1, n_epochs + 1):
        # --- 训练部分 ---
        train_t_p = model(train_t_c, *params)
        train_loss = loss_fn(train_t_p, train_t_u)

        train_losses.append(train_loss.item())

        # --- 验证部分 ---
        with torch.no_grad():
            val_t_p = model(val_t_c, *params)
            val_loss = loss_fn(val_t_p, val_t_u)
            val_losses.append(val_loss.item())

        # --- 参数更新 ---
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f"轮次 {epoch}, 训练损失 {train_loss.item():.4f},"
                  f" 验证损失 {val_loss.item():.4f}")

    return params, train_losses, val_losses


# --- 4. 执行模型训练 ---

# 初始化参数 (w 和 b)
# requires_grad=True 告诉PyTorch需要为这些张量计算梯度
params = torch.tensor([1.0, 0.0], requires_grad=True)

# 定义学习率和优化器
learning_rate = 0.01
optimizer = optim.SGD([params], lr=learning_rate)
# optimizer = optim.Adam([params], lr=learning_rate)

# 从训练循环中接收损失历史记录
final_params, train_losses, val_losses = training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    params=params,
    train_t_c=train_t_cn,
    val_t_c=val_t_cn,
    train_t_u=train_t_u,
    val_t_u=val_t_u
)
print("训练完成！")

# --- 5. 结果分析与可视化 ---

print(f"\n最终学习到的参数: {final_params.detach().numpy()}")
w, b = final_params
w_final = w.item() / 10
b_final = b.item()
print(f"学习到的最终模型公式为: F = {w_final:.4f} * C + {b_final:.4f}")
print(f"而物理学中的真实公式为: F = 1.8 * C + 32.0")

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
formula_text = f"Learned Formula:\n$F = {w_final:.4f} \\times C + {b_final:.4f}$"


# fig 是整个画布对象, axs 是一个包含两个子图坐标轴的数组 (axs[0] 和 axs[1])
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# --- 绘制第一个子图：模型拟合效果 ---
t_p = model(t_c * 0.1, *final_params)
axs[0].set_title("Celsius to Fahrenheit Conversion Model")
axs[0].set_xlabel("Temperature (Celsius)")
axs[0].set_ylabel("Temperature (Fahrenheit)")
axs[0].plot(t_c.numpy(), t_p.detach().numpy(), color='red', label="Model Prediction")
axs[0].plot(t_c.numpy(), t_u.numpy(), 'o', label="Actual Data")
axs[0].plot([], [], ' ', label=f'Learning Rate: {learning_rate}')
axs[0].text(0, 70, formula_text,
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey', boxstyle='round,pad=0.5'))
axs[0].legend()
axs[0].grid(True)

# --- 绘制第二个子图：损失曲线 ---
axs[1].set_title("Training and Validation Loss Curve")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].plot(range(len(train_losses)), train_losses, label="Training Loss")
axs[1].plot(range(len(val_losses)), val_losses, label="Validation Loss")
axs[1].plot([], [], ' ', label=f'Learning Rate: {learning_rate}')
axs[1].legend()
axs[1].grid(True) # 为子图也开启网格

# 自动调整子图布局，防止标题和标签重叠
plt.tight_layout()

# 保存整个大图
save_path = f'../output/combined_plot_{learning_rate}_{current_time}.png'
plt.savefig(save_path)
print(f"\n合并后的图表已保存至 {save_path}")

# --- 修改结束 ---
