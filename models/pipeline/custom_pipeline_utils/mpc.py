import torch

# ==============================
# 参数配置
# ==============================
B = 32                     # batch size
H = 10                     # 预测时域
action_dim = 1
num_samples = 256
num_elites = 32
num_iters = 5
u_min, u_max = -2.0, 2.0
sigma_decay = 0.9          # 方差衰减系数
sigma_min = 0.05           # 最小方差
alpha_smooth = 0.8         # 平滑系数（越大越平滑）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 动力学模型: x_{t+1} = x_t + u_t + noise
# ==============================
def dynamics(x, u):
    return x + u + 0.1 * torch.randn_like(x)

# ==============================
# 代价函数
# ==============================
def cost_function(x_seq, u_seq, x_goal=5.0):
    state_cost = (x_seq[:, :, :-1, :] - x_goal) ** 2
    control_cost = 0.1 * (u_seq ** 2)
    total_cost = state_cost.sum(dim=2) + control_cost.sum(dim=2)
    return total_cost.mean(dim=-1)  # (B, N)

# ==============================
# 带 σ衰减 + 平滑更新 的 CEM-MPC
# ==============================
def cem_mpc_step(x_init):
    B, Dx = x_init.shape
    mu = torch.zeros(B, H, action_dim, device=device)
    std = torch.ones(B, H, action_dim, device=device) * 1.5

    for it in range(num_iters):
        # 1. 采样控制序列
        eps = torch.randn(B, num_samples, H, action_dim, device=device)
        u_samples = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        u_samples = torch.clamp(u_samples, u_min, u_max)

        # 2. 模拟预测
        x = x_init.unsqueeze(1).expand(-1, num_samples, -1)
        x_seq = [x]
        for t in range(H):
            x = dynamics(x, u_samples[:, :, t, :])
            x_seq.append(x)
        x_seq = torch.stack(x_seq, dim=2)  # (B, N, H+1, Dx)

        # 3. 计算代价
        costs = cost_function(x_seq, u_samples)  # (B, N)

        # 4. 选取精英样本
        elite_idx = costs.argsort(dim=1)[:, :num_elites]
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        elites = u_samples[batch_idx, elite_idx]  # (B, K, H, Du)

        # 5. 更新分布参数
        elite_mu = elites.mean(dim=1)
        elite_std = elites.std(dim=1)

        # σ 衰减
        elite_std = torch.clamp(elite_std * sigma_decay, min=sigma_min)

        # --- 平滑更新 ---
        mu = alpha_smooth * mu + (1 - alpha_smooth) * elite_mu
        std = alpha_smooth * std + (1 - alpha_smooth) * elite_std

        best_cost = costs.min(dim=1).values.mean().item()
        print(f"Iter {it+1}: avg best cost = {best_cost:.3f}, σ={std.mean().item():.3f}")

    return mu[:, 0, :]

# ==============================
# 测试
# ==============================
torch.manual_seed(0)
x0 = torch.zeros(B, 1, device=device)
u_opt = cem_mpc_step(x0)
print("\nOptimal first controls (batch):")
print(u_opt[:10])
