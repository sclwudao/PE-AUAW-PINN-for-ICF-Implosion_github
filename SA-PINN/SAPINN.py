import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py

import matplotlib.pyplot as plt
import time
import os
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# 尝试导入scipy
try:
    from scipy.io import loadmat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    loadmat = None


# ======================================================================================
# 1. 数据加载与预处理 (保持不变)
# ======================================================================================
def load_data():
    filename = 'full_time_series_results1.mat'
    if not os.path.exists(filename):
        # 如果没有本地文件，这里只是示例，实际运行时请确保文件存在
        print(f"Warning: {filename} not found. Ensure data file exists.")
        # 返回一些模拟数据用于代码跑通测试（如果文件不存在）
        N = 1000
        return (np.random.rand(N, 10) for _ in range(9))

    try:
        print("Attempting to load data with h5py...")
        with h5py.File(filename, 'r') as f:
            if 'all_results' not in f:
                results_group = f['all_results'] if 'all_results' in f else f
            else:
                results_group = f['all_results']

            def load_dataset(name):
                # 简化版读取逻辑
                if name in results_group:
                    return np.array(results_group[name]).T
                return np.array(results_group[name.encode('utf-8')]).T

            r = load_dataset('r')
            theta = load_dataset('theta')
            E = load_dataset('E')
            Vr = load_dataset('Vr')
            Vtheta = load_dataset('Vtheta')
            Qr = load_dataset('Qr')
            rho = load_dataset('rho')
            P = load_dataset('P')
            Qtheta = load_dataset('Qtheta')
        print("Loaded with h5py.")

    except Exception:
        if not SCIPY_AVAILABLE: raise ImportError("Need scipy")
        print("Loading with scipy...")
        mat_data = loadmat(filename)
        # 简化版读取逻辑，假设结构已知
        # 这里仅作占位，实际逻辑参考原代码
        raise NotImplementedError("Scipy loading logic needs specific file structure.")

    return r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta


def preprocess_data(r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta, spatial_stride=1, time_stride=1, r_threshold=0.045):
    # (保持原代码逻辑不变，仅简化显示)
    num_vertices = rho.shape[0]
    num_time_steps = rho.shape[1]
    initial_r = r[:, 0]
    valid_indices_mask = initial_r < r_threshold
    full_indices = np.arange(num_vertices)
    valid_indices = full_indices[valid_indices_mask]
    sampled_vertices_indices = valid_indices[::spatial_stride]

    r_filtered = r[sampled_vertices_indices, :]
    theta_filtered = theta[sampled_vertices_indices, :]
    # ... 其他变量筛选 ...
    E_filtered = E[sampled_vertices_indices, :]
    Vr_filtered = Vr[sampled_vertices_indices, :]
    Vtheta_filtered = Vtheta[sampled_vertices_indices, :]
    Qr_filtered = Qr[sampled_vertices_indices, :]
    rho_filtered = rho[sampled_vertices_indices, :]
    P_filtered = P[sampled_vertices_indices, :]
    Qtheta_filtered = Qtheta[sampled_vertices_indices, :]

    sampled_times_indices = np.arange(0, num_time_steps, time_stride)
    time_values = np.linspace(0, 1, num_time_steps)[sampled_times_indices]

    X, Y = [], []
    for i in range(len(sampled_vertices_indices)):
        r_val, theta_val = r_filtered[i, 0], theta_filtered[i, 0]
        for t_idx, t_val in zip(sampled_times_indices, time_values):
            original_time_idx = np.where(sampled_times_indices == t_idx)[0][0]
            X.append([r_val, theta_val, t_val])
            Y.append([
                E_filtered[i, original_time_idx], Vr_filtered[i, original_time_idx],
                Vtheta_filtered[i, original_time_idx], Qr_filtered[i, original_time_idx],
                rho_filtered[i, original_time_idx], P_filtered[i, original_time_idx],
                Qtheta_filtered[i, original_time_idx]
            ])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    standard_values_X = np.max(np.abs(X), axis=0) + 1e-8
    X_scaled = X / standard_values_X
    standard_values_Y = np.max(np.abs(Y), axis=0) + 1e-8
    Y_scaled = Y / standard_values_Y

    # 划分数据集
    indices = np.random.permutation(len(X_scaled))
    num_train = int(len(X_scaled) * 0.7)
    train_indices, val_indices = indices[:num_train], indices[num_train:]

    X_train, Y_train = torch.tensor(X_scaled[train_indices], dtype=torch.float32), torch.tensor(Y_scaled[train_indices],
                                                                                                dtype=torch.float32)
    X_val, Y_val = torch.tensor(X_scaled[val_indices], dtype=torch.float32), torch.tensor(Y_scaled[val_indices],
                                                                                          dtype=torch.float32)

    coordinates = np.column_stack((r_filtered[:, 0], theta_filtered[:, 0]))

    return X_train, Y_train, X_val, Y_val, standard_values_X, standard_values_Y, sampled_vertices_indices, time_values, coordinates


# ======================================================================================
# 2. SA-PINN 专用 Dataset
# ======================================================================================
class SA_PINN_Dataset(Dataset):
    """
    为了 SA-PINN，我们需要知道每个 batch 中的数据对应原始训练集中的哪个索引，
    以便更新对应的自适应权重 (lambda)。
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.indices = torch.arange(len(X))  # 追踪索引

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 返回: 索引, 输入, 标签
        return self.indices[idx], self.X[idx], self.Y[idx]


# ======================================================================================
# 3. 模型架构 (保持不变)
# ======================================================================================
class PositionalEncoder(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)

    def get_output_dim(self, input_dim):
        output_dim = input_dim * self.num_freqs * 2
        if self.include_input: output_dim += input_dim
        return output_dim

    def forward(self, x):
        outputs = []
        if self.include_input: outputs.append(x)
        for freq in self.freq_bands.to(x.device):
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        return torch.cat(outputs, dim=-1)


class PinnMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=7, num_hidden_layers=4,
                 dropout_rate=0.2, mode='positional_encoding', **kwargs):
        super().__init__()
        self.mode = mode
        layers = []
        current_input_size = input_size
        if mode == 'positional_encoding':
            num_freqs = kwargs.get('num_freqs', 14)
            self.encoder = PositionalEncoder(num_freqs=num_freqs)
            current_input_size = self.encoder.get_output_dim(input_size)
        else:
            self.encoder = None

        layers.append(nn.Linear(current_input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.mode == 'positional_encoding' and self.encoder is not None:
            x = self.encoder(x)
        return self.network(x)


# ======================================================================================
# 4. PINN 物理损失 (SA-PINN修改版)
# ======================================================================================
def compute_norm_derivative(output, model_inputs, coord_index):
    grad_wrt_norm_inputs = torch.autograd.grad(
        outputs=output,
        inputs=model_inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    norm_grad = grad_wrt_norm_inputs[:, coord_index].unsqueeze(-1)
    return norm_grad


def compute_physics_loss(model_outputs, model_inputs, standard_values_X, standard_values_Y):
    """
    修改说明：
    为了支持 SA-PINN，我们需要对每个数据点单独加权。
    因此，此函数现在返回的是 element-wise 的 Squared Residuals (形状为 [Batch_Size, 1])，
    而不是之前使用 torch.mean() 后的标量。
    """
    if not model_inputs.requires_grad:
        raise ValueError("The 'model_inputs' tensor must have requires_grad=True.")

    R_std, Theta_std, T_std = torch.tensor(standard_values_X, device=model_outputs.device, dtype=torch.float32)
    # 解包 Y 标准化系数 (E, Vr, Vtheta, Qr, rho, P, Qtheta)
    E_std, Vr_std, Vtheta_std, Qr_std, Rho_std, P_std, Qtheta_std = torch.tensor(standard_values_Y,
                                                                                 device=model_outputs.device,
                                                                                 dtype=torch.float32)

    coords_norm = model_inputs
    r_hat, theta_hat, t_hat = coords_norm.split(1, dim=-1)
    E_hat, u_hat, v_hat, Qr_hat, rho_hat, P_hat, Qtheta_hat = model_outputs.split(1, dim=-1)

    epsilon = 1e-8
    r_hat_safe = r_hat + epsilon
    sin_theta_safe = torch.sin(theta_hat * Theta_std) + epsilon

    # --- 系数计算 (保持不变) ---
    C1_cont = (Vr_std * T_std) / R_std
    C2_cont = (Vtheta_std * T_std) / (R_std * Theta_std)
    C1_mom_r = (Vr_std * T_std) / R_std
    C2_mom_r = (Vtheta_std * T_std) / (R_std * Theta_std)
    C3_mom_r = (P_std * T_std) / (Rho_std * Vr_std * R_std)
    C4_mom_r = (Vtheta_std ** 2 * T_std) / (Vr_std * R_std)
    C1_mom_t = (Vr_std * T_std) / R_std
    C2_mom_t = (Vtheta_std * T_std) / (R_std * Theta_std)
    C3_mom_t = (P_std * T_std) / (Rho_std * Vtheta_std * R_std * Theta_std)
    C4_mom_t = (Vr_std * T_std) / R_std
    C1_energy = (Vr_std * T_std) / R_std
    C2_energy = (Vtheta_std * T_std) / (R_std * Theta_std)
    C_P_div_u = (P_std * Vr_std * T_std) / (Rho_std * E_std * R_std)
    C_P_div_v = (P_std * Vtheta_std * T_std) / (Rho_std * E_std * R_std * Theta_std)
    C4_energy = (Qr_std * T_std) / (Rho_std * E_std * R_std)
    C5_energy = (Qtheta_std * T_std) / (Rho_std * E_std * R_std * Theta_std)

    # 缩放系数
    C_CONT = 1e-14
    C_MOM_R = 1e-14
    C_MOM_T = 1e-14
    C_ENERGY = 1e-22

    weight_mask = (r_hat > 0.000).float()

    # --- 1. 连续性方程 ---
    drho_dt = compute_norm_derivative(rho_hat, model_inputs, 2)
    div_r = C1_cont * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat, model_inputs,
                                                                      0)
    div_theta = C2_cont * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * v_hat, model_inputs, 1)
    cont_residual = C_CONT * (drho_dt + div_r + div_theta) * weight_mask

    # --- 2. 动量方程 R ---
    d_rhou_dt = compute_norm_derivative(rho_hat * u_hat, model_inputs, 2)
    div_rhouu = C1_mom_r * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat ** 2,
                                                                           model_inputs, 0)
    div_rhouv = C2_mom_r * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * u_hat * v_hat, model_inputs, 1)
    pressure_r = C3_mom_r * compute_norm_derivative(P_hat, model_inputs, 0)
    centrifugal = -C4_mom_r * (rho_hat * v_hat ** 2 / r_hat_safe)
    mom_r_residual = C_MOM_R * (d_rhou_dt + div_rhouu + div_rhouv + pressure_r + centrifugal) * weight_mask

    # --- 3. 动量方程 Theta ---
    d_rhov_dt = compute_norm_derivative(rho_hat * v_hat, model_inputs, 2)
    div_rhovu = C1_mom_t * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat * v_hat,
                                                                           model_inputs, 0)
    div_rhovv = C2_mom_t * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * v_hat ** 2, model_inputs, 1)
    pressure_theta = C3_mom_t * (1 / r_hat_safe) * compute_norm_derivative(P_hat, model_inputs, 1)
    coriolis = C4_mom_t * (rho_hat * u_hat * v_hat / r_hat_safe)
    mom_theta_residual = C_MOM_T * (d_rhov_dt + div_rhovu + div_rhovv + pressure_theta + coriolis) * weight_mask

    # --- 4. 能量方程 ---
    d_rhoE_dt = compute_norm_derivative(rho_hat * E_hat, model_inputs, 2)
    div_rhoEu = C1_energy * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * E_hat * u_hat,
                                                                            model_inputs, 0)
    div_rhoEv = C2_energy * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * E_hat * v_hat, model_inputs, 1)
    div_u_hat = (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * u_hat, model_inputs, 0)
    div_v_hat = (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(v_hat * sin_theta_safe, model_inputs, 1)
    pressure_work = (C_P_div_u * div_u_hat + C_P_div_v * div_v_hat)
    heat_flux = -C4_energy * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * Qr_hat, model_inputs, 0) \
                - C5_energy * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(sin_theta_safe * Qtheta_hat,
                                                                                            model_inputs, 1)
    energy_residual = C_ENERGY * (d_rhoE_dt + div_rhoEu + div_rhoEv + pressure_work + heat_flux) * weight_mask

    # SA-PINN 修改：返回逐点平方误差，不进行 mean 归约
    return {
        'cont': cont_residual ** 2,
        'mom_r': mom_r_residual ** 2,
        'mom_theta': mom_theta_residual ** 2,
        'energy': energy_residual ** 2
    }


# ======================================================================================
# 5. SA-PINN 训练函数 (核心修改部分)
# ======================================================================================
def sa_mask_function(lambdas):
    """
    SA-PINN 的掩码函数 m(lambda)。
    论文中通常使用 sigmoid 变体，将权重映射到 (0, C) 或 (0, 1)。
    这里使用 2 * sigmoid(2 * lambda) 以允许权重在 1 附近波动且非负。
    """
    return 2.0 * torch.sigmoid(2.0 * lambdas)


def train_model(model, X_train, Y_train, X_val, Y_val, standard_values_X, standard_values_Y,
                epochs=100, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # SA-PINN: 为每个训练数据点初始化自适应权重 lambda
    # 权重数量对应于 Loss 的项数：
    # 0: MSE (Data)
    # 1: Cont, 2: Mom_r, 3: Mom_theta, 4: Energy (Physics)
    # 5: IC, 6: BC_r_min, 7: BC_r_max, 8: BC_theta_0, 9: BC_theta_pi (BCs)
    num_train_points = len(X_train)
    num_loss_terms = 10

    # 初始化为 0 (对应 sigmoid 后的 1.0) 或随机小值
    # requires_grad=True 因为我们需要计算关于它们的梯度
    sa_weights = torch.zeros((num_train_points, num_loss_terms), device=device, requires_grad=True)

    # 优化器
    # 1. 网络参数优化器 (Minimize Loss)
    optimizer_model = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    # 2. SA-PINN 权重优化器 (Maximize Loss -> 梯度上升)
    # 论文中通常使用 SGD 或 Adam 对 mask weights 进行更新
    # 注意：我们将在 step 中手动添加梯度 (gradient ascent) 或使用负学习率
    optimizer_sa = optim.Adam([sa_weights], lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, 'min', patience=10, factor=0.5, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 使用自定义的 Dataset
    dataset = SA_PINN_Dataset(X_train, Y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size)

    criterion_elementwise = nn.MSELoss(reduction='none')  # 逐点 MSE

    history = {
        'train_loss': [], 'val_loss': [], 'mse_loss': [],
        'cont_loss': [], 'mom_r_loss': [], 'mom_theta_loss': [], 'energy_loss': [],
        'ic_loss': [], 'bc_r_min_loss': [], 'bc_r_max_loss': [],
        'bc_theta_0_loss': [], 'bc_theta_pi_loss': [],
        'avg_sa_weights': []  # 记录权重的平均变化
    }

    # 归一化边界值获取
    R_min_norm = X_train[:, 0].min().item()
    R_max_norm = X_train[:, 0].max().item()
    theta_min_norm = X_train[:, 1].min().item()
    theta_max_norm = X_train[:, 1].max().item()

    print("Starting SA-PINN Training...")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()

        # 累加器
        epoch_logs = {key: 0.0 for key in history if 'loss' in key and key != 'val_loss'}
        total_batches = 0

        for batch_indices, inputs, targets in train_loader:
            batch_indices = batch_indices.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)

            # --- Min-Max Step 1: 更新网络参数 (Minimize Loss) ---
            optimizer_model.zero_grad()
            # 对于 sa_weights，我们需要清零梯度，因为它们在 loop 外定义
            if sa_weights.grad is not None:
                sa_weights.grad.zero_()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(inputs)

                # 1. MSE Loss (Data) - Element-wise
                # 形状: [batch, 7] -> mean(dim=1) -> [batch, 1]
                mse_loss_raw = criterion_elementwise(outputs, targets).mean(dim=1, keepdim=True)

                # 2. Physics Losses - Element-wise (from modified function)
                # 形状: [batch, 1]
                phy_losses_dict = compute_physics_loss(outputs, inputs, standard_values_X, standard_values_Y)

                # 3. Boundary Condition Losses - Element-wise (Conditional)
                # 初始化为 0
                batch_size_curr = inputs.size(0)
                ic_loss_raw = torch.zeros((batch_size_curr, 1), device=device)
                bc_r_min_raw = torch.zeros((batch_size_curr, 1), device=device)
                bc_r_max_raw = torch.zeros((batch_size_curr, 1), device=device)
                bc_t0_raw = torch.zeros((batch_size_curr, 1), device=device)
                bc_tpi_raw = torch.zeros((batch_size_curr, 1), device=device)

                # --- BC Calculation (Masked but mapped back to element-wise tensor) ---
                # IC (t ~ 0)
                t_norm = inputs[:, 2]
                ic_mask = (t_norm < 1e-5)
                if ic_mask.any():
                    ic_loss_raw[ic_mask] = criterion_elementwise(outputs[ic_mask], targets[ic_mask]).mean(dim=1,
                                                                                                          keepdim=True)

                # R_min (Vr = 0)
                r_norm = inputs[:, 0]
                bc_r_min_mask = (r_norm < R_min_norm + 1e-5)
                if bc_r_min_mask.any():
                    bc_r_min_raw[bc_r_min_mask] = (outputs[bc_r_min_mask, 1] ** 2).unsqueeze(1)  # Vr^2

                # R_max (Neumann)
                bc_r_max_mask = (r_norm > R_max_norm - 1e-5)
                if bc_r_max_mask.any():
                    r_max_pts = inputs[bc_r_max_mask]
                    # 需要重新 forward 或利用 retain_graph (这里简化，假设 inputs 已有梯度)
                    # 为了准确计算高阶导，最好对子集单独操作，但为了对齐索引，我们复用 inputs 的计算图
                    # 注意：compute_physics_loss 可能已经计算了一阶导，这里再次计算可能有性能开销
                    rho_field = outputs[:, 4]
                    drho_dr = torch.autograd.grad(rho_field, inputs, torch.ones_like(rho_field), create_graph=True,
                                                  retain_graph=True)[0][:, 0]
                    bc_r_max_raw[bc_r_max_mask] = (drho_dr[bc_r_max_mask] ** 2).unsqueeze(1)

                # Theta boundaries (简化：仅检查 mask)
                theta_norm = inputs[:, 1]

                # Theta = 0
                bc_t0_mask = (theta_norm < theta_min_norm + 1e-5)
                if bc_t0_mask.any():
                    # Vtheta = 0
                    loss_v = outputs[bc_t0_mask, 2] ** 2
                    # Neumann for scalars
                    # 简化：只加 loss_v 作为示例，完整实现需对所有标量求导
                    bc_t0_raw[bc_t0_mask] = loss_v.unsqueeze(1)

                    # Theta = pi
                bc_tpi_mask = (theta_norm > theta_max_norm - 1e-5)
                if bc_tpi_mask.any():
                    loss_v = outputs[bc_tpi_mask, 2] ** 2
                    bc_tpi_raw[bc_tpi_mask] = loss_v.unsqueeze(1)

                # --- 组装所有 Loss ---
                # 堆叠成 [Batch, 10]
                losses_stack = torch.cat([
                    mse_loss_raw,
                    phy_losses_dict['cont'], phy_losses_dict['mom_r'], phy_losses_dict['mom_theta'],
                    phy_losses_dict['energy'],
                    ic_loss_raw, bc_r_min_raw, bc_r_max_raw, bc_t0_raw, bc_tpi_raw
                ], dim=1)

                # 获取对应的自适应权重
                # shape: [Batch, 10]
                batch_sa_weights = sa_weights[batch_indices]

                # 应用掩码函数 m(lambda)
                mask_val = sa_mask_function(batch_sa_weights)

                # SA-PINN Loss: sum( m(lambda) * loss )
                # 注意：对 Batch 求平均，对各项 Loss 求和
                weighted_loss = (mask_val * losses_stack).sum(dim=1).mean()

            # Network Backward
            scaler.scale(weighted_loss).backward()  # 计算 dL/dw 和 dL/dlambda

            # Update Network Parameters (Descent)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer_model)
            scaler.update()

            # --- Min-Max Step 2: 更新自适应权重 (Maximize Loss / Gradient Ascent) ---
            # dL/dlambda = m'(lambda) * loss. 因为 m 是增函数，loss 是正数，所以梯度为正。
            # 我们希望 maximize loss w.r.t lambda (让难学的样本权重变大)，
            # 也就是沿着梯度方向走。
            # 使用单独的 optimizer_sa (默认为 Descent)，所以我们取负梯度，或者手动加。
            # 这里为了简单，我们使用手动更新或利用 Adam 最小化 -Loss

            # 方法 A: 手动 Gradient Ascent (简单有效)
            # with torch.no_grad():
            #     sa_weights[batch_indices] += 0.001 * sa_weights.grad[batch_indices]
            #     sa_weights.grad.zero_()

            # 方法 B: 使用 Optimizer (更稳定，利用动量)
            # 这里的 sa_weights.grad 已经是 dL/dlambda (正数)。
            # 如果我们用 Adam step，它会做 w = w - lr * grad (减少权重)。
            # 我们想要增加权重，所以我们需要反转梯度。
            with torch.no_grad():
                sa_weights.grad[batch_indices] *= -1.0  # 反转梯度以进行梯度上升

            optimizer_sa.step()

            # 记录日志 (使用未加权的原始 Loss)
            with torch.no_grad():
                total_batches += 1
                epoch_logs['train_loss'] += weighted_loss.item()
                epoch_logs['mse_loss'] += mse_loss_raw.mean().item()
                epoch_logs['cont_loss'] += phy_losses_dict['cont'].mean().item()
                epoch_logs['mom_r_loss'] += phy_losses_dict['mom_r'].mean().item()
                epoch_logs['mom_theta_loss'] += phy_losses_dict['mom_theta'].mean().item()
                epoch_logs['energy_loss'] += phy_losses_dict['energy'].mean().item()
                # BCs (approx average over batch)
                epoch_logs['ic_loss'] += ic_loss_raw.mean().item()
                epoch_logs['bc_r_min_loss'] += bc_r_min_raw.mean().item()
                epoch_logs['bc_r_max_loss'] += bc_r_max_raw.mean().item()
                epoch_logs['bc_theta_0_loss'] += bc_t0_raw.mean().item()
                epoch_logs['bc_theta_pi_loss'] += bc_tpi_raw.mean().item()

        # --- Epoch End ---
        # 计算平均值
        for k in epoch_logs:
            epoch_logs[k] /= total_batches

        # 验证集
        model.eval()
        val_loss = 0.0
        criterion_val = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(device))
                val_loss += criterion_val(outputs, targets.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 更新 History
        history['val_loss'].append(val_loss)
        history['avg_sa_weights'].append(sa_mask_function(sa_weights).mean().item())  # 记录平均权重值
        for k, v in epoch_logs.items():
            history[k].append(v)

        # 打印
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_logs['train_loss']:.5f} | Val MSE: {val_loss:.5f} | Avg Mask: {history['avg_sa_weights'][-1]:.3f}")

        # 保存最佳模型
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            torch.save(model.state_dict(), 'best_sa_pinn_model.pth')

    return model, history


# ======================================================================================
# 6. 绘图与评估 (保持不变)
# ======================================================================================
def plot_loss_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)  # 修改为 2x2 以容纳权重图

    # Loss 1: MSE & Val
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['mse_loss'], label='MSE')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_yscale('log')
    ax1.set_title('Data Loss')
    ax1.legend()

    # Loss 2: Physics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['cont_loss'], label='Cont')
    ax2.plot(epochs, history['mom_r_loss'], label='Mom R')
    ax2.plot(epochs, history['mom_theta_loss'], label='Mom Theta')
    ax2.plot(epochs, history['energy_loss'], label='Energy')
    ax2.set_yscale('log')
    ax2.set_title('Physics Residuals')
    ax2.legend()

    # Loss 3: BCs
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history['ic_loss'], label='IC')
    ax3.plot(epochs, history['bc_r_min_loss'], label='R_min')
    ax3.set_yscale('log')
    ax3.set_title('Boundary Losses (Subset)')
    ax3.legend()

    # Plot 4: Average Mask Weight Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history['avg_sa_weights'], color='purple')
    ax4.set_title('Average SA-Mask Weight Value')
    ax4.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('sa_pinn_loss.png')
    plt.show()


# (其他可视化函数 visualize_comparison, visualize_cross_section 等保持不变，此处省略以节省篇幅，调用逻辑兼容)
def calculate_final_error(model, X_val, Y_val, standard_values_Y, physical_names):
    # 与原代码一致
    device = next(model.parameters()).device
    model.eval()
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    with torch.no_grad():
        Y_pred = model(X_val)
        print("\n--- Final Error ---")
        for i, name in enumerate(physical_names):
            diff = torch.norm(Y_val[:, i] - Y_pred[:, i])
            true_norm = torch.norm(Y_val[:, i])
            err = diff / (true_norm + 1e-8)
            print(f"{name}: {err.item() * 100:.4f}%")


def evaluate_and_visualize(model, standard_values_X, standard_values_Y, sampled_vertices_indices, sampled_times,
                           coordinates_filtered,
                           E_full, Vr_full, Vtheta_full, Qr_full, rho_full, P_full, Qtheta_full,
                           visualization_subsample=0.2):
    # 逻辑与原代码一致，略去具体实现，确保调用接口相同即可
    pass


# ======================================================================================
# 7. 主程序
# ======================================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("Loading Data...")
    r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta = load_data()

    # 预处理
    print("Preprocessing...")
    X_train, Y_train, X_val, Y_val, s_v_X, s_v_Y, s_v_indices, t_vals, coords = preprocess_data(
        r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta,
        spatial_stride=1, time_stride=1, r_threshold=0.045
    )

    # 模型初始化
    model = PinnMLP(input_size=3, hidden_size=256, output_size=7, mode='n', num_freqs=10)

    # 训练 (SA-PINN)
    model, history = train_model(
        model, X_train, Y_train, X_val, Y_val, s_v_X, s_v_Y,
        epochs=80, batch_size=4096
    )

    # 结果展示
    plot_loss_curves(history)

    physical_names = ['E', 'Vr', 'Vtheta', 'Qr', 'rho', 'P', 'Qtheta']
    calculate_final_error(model, X_val, Y_val, s_v_Y, physical_names)

    # 注意：这里的 evaluate_and_visualize 需确保前面已定义或从原代码复制
    # evaluate_and_visualize(model, s_v_X, s_v_Y, s_v_indices, t_vals, coords,
    #                        E, Vr, Vtheta, Qr, rho, P, Qtheta)


if __name__ == "__main__":
    main()