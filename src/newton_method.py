import torch
import matplotlib.pyplot as plt
import os
import matplotlib
from datetime import datetime

# --- 1. 环境与数据准备 ---

plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用全部数据进行训练，以展示牛顿法的威力
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0,
                    -9.0, -5.5, 1.0, 4.5, 9.0, 12.5, 17.0, 19.5, 24.0, 27.0, 30.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4,
                    14.1, 18.3, 33.5, 38.6, 46.5, 53.6, 61.3, 64.3, 73.1, 80.0, 84.4])
# 数据归一化
t_cn = 0.1 * t_c

# --- 2. 定义模型与损失函数 ---

def model(t_c_normalized, w, b):
    return w * t_c_normalized + b

def loss_fn(t_p, t_u):
    return ((t_p - t_u)**2).mean()

# --- 3. 牛顿法训练 ---

params = torch.tensor([1.0, 0.0], requires_grad=True)
n_epochs = 5 # 牛顿法收敛极快，5轮足矣
loss_history = []

print("开始使用牛顿法训练...")
for epoch in range(1, n_epochs + 1):
    # 计算损失和一阶梯度 (g)
    t_p = model(t_cn, *params)
    loss = loss_fn(t_p, t_u)
    loss_history.append(loss.item())
    loss.backward()
    g = params.grad.clone()

    # 手动计算二阶导数 (海森矩阵 H)
    H = torch.zeros(2, 2)
    H[0, 0] = torch.mean(2 * t_cn**2)
    H[1, 1] = 2.0
    H[0, 1] = H[1, 0] = torch.mean(2 * t_cn) # 海森矩阵是对称的

    # 手动更新参数，清空梯度
    params.grad.zero_()
    with torch.no_grad():
        update_step = torch.inverse(H) @ g
        params -= update_step

    print(f"轮次 {epoch}, 损失 {loss.item():.4f}, "
          f"参数 w: {params[0].item():.4f}, b: {params[1].item():.4f}")

print("\n训练完成！")

# --- 4. 结果分析与可视化 ---

w_final = params[0].item() / 10
b_final = params[1].item()
print(f"学习到的最终模型公式为: F = {w_final:.4f} * C + {b_final:.4f}")
print(f"而物理学中的真实公式为: F = 1.8 * C + 32.0")

# 准备绘图
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
formula_text = f"Learned Formula (Newton's Method):\n$F = {w_final:.4f} \\times C + {b_final:.4f}$"

# 创建一个包含1行2列子图的大图
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# --- 绘制第一个子图：模型拟合效果 ---
t_p_final = model(t_cn, *params)
axs[0].set_title("Newton's Method - Model Fit")
axs[0].set_xlabel("Temperature (Celsius)")
axs[0].set_ylabel("Temperature (Fahrenheit)")
axs[0].plot(t_c.numpy(), t_p_final.detach().numpy(), color='red', label="Model Prediction")
axs[0].plot(t_c.numpy(), t_u.numpy(), 'o', label="Actual Data")
axs[0].text(0, 70, formula_text, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey', boxstyle='round,pad=0.5'))
axs[0].legend()
axs[0].grid(True)

# --- 绘制第二个子图：损失曲线 ---
axs[1].set_title("Loss Curve (Newton's Method)")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].plot(range(1, n_epochs + 1), loss_history, 'o-', label="Loss") # 用点线图更清晰
axs[1].legend()
axs[1].grid(True)
axs[1].set_xticks(range(1, n_epochs + 1)) # 设置x轴刻度为整数

# 调整布局并保存
plt.tight_layout()

plt.savefig(f'../output/newton_method_combined_{current_time}.png')
print(f"\n合并后的图表已保存至../output/newton_method_combined_{current_time}.png")