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

# 尝试导入scipy，如果不可用则设为None
try:
    from scipy.io import loadmat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    loadmat = None


# 1. 加载MATLAB生成的数据
def load_data():
    filename = 'full_time_series_results1.mat'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found in current directory")

    # 首先尝试用h5py读取
    try:
        print("Attempting to load data with h5py...")
        with h5py.File(filename, 'r') as f:
            if 'all_results' not in f:
                if b'all_results' in f:
                    results_group = f[b'all_results']
                else:
                    available_keys = list(f.keys())
                    raise KeyError(f"'all_results' struct not found in {filename}. Available keys: {available_keys}")
            else:
                results_group = f['all_results']

            print(f"Keys in 'all_results' group: {list(results_group.keys())}")

            def load_dataset(name):
                if name in results_group:
                    dataset_ref = results_group[name][()]
                    if isinstance(dataset_ref, h5py.Reference):
                        dataset = f[dataset_ref]
                    elif dataset_ref.size > 0 and isinstance(dataset_ref[0], h5py.Reference):
                        dataset = f[dataset_ref[0]]
                    else:
                        dataset = results_group[name]
                    return np.array(dataset).T
                name_bytes = name.encode('utf-8')
                if name_bytes in results_group:
                    dataset_ref = results_group[name_bytes][()]
                    if isinstance(dataset_ref, h5py.Reference):
                        dataset = f[dataset_ref]
                    elif dataset_ref.size > 0 and isinstance(dataset_ref[0], h5py.Reference):
                        dataset = f[dataset_ref[0]]
                    else:
                        dataset = results_group[name_bytes]
                    return np.array(dataset).T
                available_keys = list(results_group.keys())
                raise KeyError(f"Field '{name}' not found in all_results struct. Available keys: {available_keys}")

            r = load_dataset('r')
            theta = load_dataset('theta')
            E = load_dataset('E')
            Vr = load_dataset('Vr')
            Vtheta = load_dataset('Vtheta')
            Qr = load_dataset('Qr')
            rho = load_dataset('rho')
            P = load_dataset('P')
            Qtheta = load_dataset('Qtheta')

        print("Successfully loaded all physical quantities from 'all_results' struct using h5py")

    except (OSError, KeyError, ValueError) as h5py_error:
        print(f"h5py loading failed: {h5py_error}")

        # 如果h5py读取失败，尝试使用scipy
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not available. Please install scipy to read .mat files: pip install scipy")

        print("Attempting to load data with scipy...")
        try:
            mat_data = loadmat(filename)
            print(f"Available keys in .mat file: {list(mat_data.keys())}")

            # 查找包含我们数据的结构体
            all_results_key = None
            for key in mat_data.keys():
                if key.startswith('__') and key.endswith('__'):  # 跳过元数据
                    continue
                if isinstance(mat_data[key], np.ndarray) and mat_data[key].dtype.names is not None:
                    # 这是一个结构体数组
                    all_results_key = key
                    break

            if all_results_key is None:
                # 如果没有找到结构体，尝试直接读取变量
                available_fields = {}
                for key in mat_data.keys():
                    if not (key.startswith('__') and key.endswith('__')):
                        available_fields[key] = mat_data[key].shape
                raise KeyError(f"No structured array found in {filename}. Available fields: {available_fields}")

            results_struct = mat_data[all_results_key]
            print(f"Using structured array from key: '{all_results_key}'")
            print(f"Fields in structure: {results_struct.dtype.names}")

            def load_scipy_dataset(name):
                if name in results_struct.dtype.names:
                    data = results_struct[name][0, 0]  # MATLAB结构体通常是1x1的cell数组
                    # 确保数据是2D数组
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    return data
                # 尝试不同的名称变体
                name_variants = [name, name.lower(), name.upper()]
                for variant in name_variants:
                    if variant in results_struct.dtype.names:
                        data = results_struct[variant][0, 0]
                        if data.ndim == 1:
                            data = data.reshape(-1, 1)
                        return data
                available_fields = list(results_struct.dtype.names)
                raise KeyError(f"Field '{name}' not found in structure. Available fields: {available_fields}")

            r = load_scipy_dataset('r')
            theta = load_scipy_dataset('theta')
            E = load_scipy_dataset('E')
            Vr = load_scipy_dataset('Vr')
            Vtheta = load_scipy_dataset('Vtheta')
            Qr = load_scipy_dataset('Qr')
            rho = load_scipy_dataset('rho')
            P = load_scipy_dataset('P')
            Qtheta = load_scipy_dataset('Qtheta')

            print("Successfully loaded all physical quantities using scipy")

        except Exception as scipy_error:
            raise RuntimeError(f"Both h5py and scipy failed to load the file. "
                               f"h5py error: {h5py_error}, scipy error: {scipy_error}")

    print(f"Data shapes: r={r.shape}, theta={theta.shape}, E={E.shape}, Vr={Vr.shape}, "
          f"Vtheta={Vtheta.shape}, Qr={Qr.shape}, rho={rho.shape}, P={P.shape}, Qtheta={Qtheta.shape}")

    return r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta


# 2. 数据预处理函数
def preprocess_data(r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta, spatial_stride=1, time_stride=1, r_threshold=0.045):
    """
    数据预处理函数，用于 (r, theta, t) -> (fields) 的映射。
    - 每个时空点都是一个独立的样本。
    """
    num_vertices = rho.shape[0]
    num_time_steps = rho.shape[1]

    # 根据 r < r_threshold 筛选空间点
    initial_r = r[:, 0]
    valid_indices_mask = initial_r < r_threshold

    # 结合空间下采样 stride
    full_indices = np.arange(num_vertices)
    valid_indices = full_indices[valid_indices_mask]
    sampled_vertices_indices = valid_indices[::spatial_stride]

    if len(sampled_vertices_indices) == 0:
        raise ValueError(f"No data points found with r < {r_threshold}. Please check the threshold or data.")

    print(f"--- Data Filtering ---")
    print(f"Original number of spatial points: {num_vertices}")
    print(f"Number of points with r < {r_threshold}: {len(valid_indices)}")
    print(f"Number of points after applying spatial_stride={spatial_stride}: {len(sampled_vertices_indices)}")
    print(f"----------------------")

    # 使用筛选后的索引来提取数据
    r_filtered = r[sampled_vertices_indices, :]
    theta_filtered = theta[sampled_vertices_indices, :]
    E_filtered = E[sampled_vertices_indices, :]
    Vr_filtered = Vr[sampled_vertices_indices, :]
    Vtheta_filtered = Vtheta[sampled_vertices_indices, :]
    Qr_filtered = Qr[sampled_vertices_indices, :]
    rho_filtered = rho[sampled_vertices_indices, :]
    P_filtered = P[sampled_vertices_indices, :]
    Qtheta_filtered = Qtheta[sampled_vertices_indices, :]

    # 时间采样保持不变
    sampled_times_indices = np.arange(0, num_time_steps, time_stride)
    time_values = np.linspace(0, 1, num_time_steps)[sampled_times_indices]

    X, Y = [], []

    # 遍历所有采样点和采样时间
    for i in range(len(sampled_vertices_indices)):
        r_val, theta_val = r_filtered[i, 0], theta_filtered[i, 0]
        for t_idx, t_val in zip(sampled_times_indices, time_values):
            original_time_idx = np.where(sampled_times_indices == t_idx)[0][0]
            X.append([r_val, theta_val, t_val])
            Y.append([
                E_filtered[i, original_time_idx], Vr_filtered[i, original_time_idx],
                Vtheta_filtered[i, original_time_idx],
                Qr_filtered[i, original_time_idx], rho_filtered[i, original_time_idx], P_filtered[i, original_time_idx],
                Qtheta_filtered[i, original_time_idx]
            ])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"Generated data: X shape={X.shape}, Y shape={Y.shape}")

    # 归一化
    standard_values_X = np.max(np.abs(X), axis=0) + 1e-8
    X_scaled = X / standard_values_X

    standard_values_Y = np.max(np.abs(Y), axis=0) + 1e-8
    Y_scaled = Y / standard_values_Y

    print(f"Standard values for X (r, theta, t): {standard_values_X}")
    print(f"Standard values for Y (E, Vr, Vtheta, Qr, rho, P, Qtheta): {standard_values_Y}")

    # 划分数据集
    indices = np.random.permutation(len(X_scaled))
    num_train = int(len(X_scaled) * 0.7)
    train_indices, val_indices = indices[:num_train], indices[num_train:]

    X_train, Y_train = torch.tensor(X_scaled[train_indices], dtype=torch.float32), torch.tensor(Y_scaled[train_indices],
                                                                                                dtype=torch.float32)
    X_val, Y_val = torch.tensor(X_scaled[val_indices], dtype=torch.float32), torch.tensor(Y_scaled[val_indices],
                                                                                          dtype=torch.float32)

    print(
        f"Preprocessed data shapes: X_train {X_train.shape}, Y_train {Y_train.shape}, X_val {X_val.shape}, Y_val {Y_val.shape}")

    # 返回筛选后的坐标和索引
    coordinates = np.column_stack((r_filtered[:, 0], theta_filtered[:, 0]))
    return X_train, Y_train, X_val, Y_val, standard_values_X, standard_values_Y, sampled_vertices_indices, time_values, coordinates


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0, std=0.01)


# ======================================================================================
# 3. 模型架构
# ======================================================================================

class PositionalEncoder(nn.Module):
    """
    对输入坐标进行位置编码。
    """

    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = True
        if self.log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1., 2. ** (num_freqs - 1), num_freqs)

    def get_output_dim(self, input_dim):
        output_dim = input_dim * self.num_freqs * 2
        if self.include_input:
            output_dim += input_dim
        return output_dim

    def forward(self, x):
        outputs = []
        if self.include_input:
            outputs.append(x)
        for freq in self.freq_bands:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        return torch.cat(outputs, dim=-1)


class PinnMLP(nn.Module):
    """
    一个统一的MLP模型，可以通过 mode 参数选择不同的架构。
    - 'basic': 原始的ReLU网络
    - 'positional_encoding': 带位置编码的ReLU网络
    """

    def __init__(self, input_size, hidden_size=256, output_size=7, num_hidden_layers=4,
                 dropout_rate=0.2, mode='positional_encoding', **kwargs):
        super().__init__()
        self.mode = mode
        layers = []

        if mode == 'basic' or mode == 'positional_encoding':
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
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'basic' or 'positional_encoding'.")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.mode == 'positional_encoding' and self.encoder is not None:
            x = self.encoder(x)
        return self.network(x)


# 4. PINN 模块
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
    if not model_inputs.requires_grad:
        raise ValueError("The 'model_inputs' tensor must have requires_grad=True.")

    R_std, Theta_std, T_std = torch.tensor(standard_values_X, device=model_outputs.device, dtype=torch.float32)
    E_std, Vr_std, Vtheta_std, Qr_std, Rho_std, P_std, Qtheta_std = torch.tensor(standard_values_Y,
                                                                                 device=model_outputs.device,
                                                                                 dtype=torch.float32)

    coords_norm = model_inputs
    r_hat, theta_hat, t_hat = coords_norm.split(1, dim=-1)
    E_hat, u_hat, v_hat, Qr_hat, rho_hat, P_hat, Qtheta_hat = model_outputs.split(1, dim=-1)

    epsilon = 1e-8
    r_hat_safe = r_hat + epsilon
    sin_theta_safe = torch.sin(theta_hat * Theta_std) + epsilon

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

    C_CONT = 1e-14
    C_MOM_R = 1e-14
    C_MOM_T = 1e-14
    C_ENERGY = 1e-22

    weight_mask = (r_hat > 0.000).float()

    drho_dt = compute_norm_derivative(rho_hat, model_inputs, 2)
    div_r = C1_cont * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat, model_inputs,
                                                                      0)
    div_theta = C2_cont * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * v_hat, model_inputs, 1)
    cont_residual = C_CONT * (drho_dt + div_r + div_theta) * weight_mask

    d_rhou_dt = compute_norm_derivative(rho_hat * u_hat, model_inputs, 2)
    div_rhouu = C1_mom_r * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat ** 2,
                                                                           model_inputs, 0)
    div_rhouv = C2_mom_r * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * u_hat * v_hat, model_inputs, 1)
    pressure_r = C3_mom_r * compute_norm_derivative(P_hat, model_inputs, 0)
    centrifugal = -C4_mom_r * (rho_hat * v_hat ** 2 / r_hat_safe)
    mom_r_residual = C_MOM_R * (d_rhou_dt + div_rhouu + div_rhouv + pressure_r + centrifugal) * weight_mask

    d_rhov_dt = compute_norm_derivative(rho_hat * v_hat, model_inputs, 2)
    div_rhovu = C1_mom_t * (1 / r_hat_safe ** 2) * compute_norm_derivative(r_hat_safe ** 2 * rho_hat * u_hat * v_hat,
                                                                           model_inputs, 0)
    div_rhovv = C2_mom_t * (1 / (r_hat_safe * sin_theta_safe)) * compute_norm_derivative(
        sin_theta_safe * rho_hat * v_hat ** 2, model_inputs, 1)
    pressure_theta = C3_mom_t * (1 / r_hat_safe) * compute_norm_derivative(P_hat, model_inputs, 1)
    coriolis = C4_mom_t * (rho_hat * u_hat * v_hat / r_hat_safe)
    mom_theta_residual = C_MOM_T * (d_rhov_dt + div_rhovu + div_rhovv + pressure_theta + coriolis) * weight_mask

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

    loss_cont = torch.mean(cont_residual ** 2)
    loss_mom_r = torch.mean(mom_r_residual ** 2)
    loss_mom_theta = torch.mean(mom_theta_residual ** 2)
    loss_energy = torch.mean(energy_residual ** 2)

    return {
        'cont': loss_cont, 'mom_r': loss_mom_r,
        'mom_theta': loss_mom_theta, 'energy': loss_energy
    }


# 5. 训练函数
def train_model(model, X_train, Y_train, X_val, Y_val, standard_values_X, standard_values_Y,
                epochs=100, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 可学习权重
    log_vars = [torch.zeros((1,), requires_grad=True, device=device) for _ in range(5)]
    log_var_ic = torch.zeros((1,), requires_grad=True, device=device)
    log_var_bc_r_min = torch.zeros((1,), requires_grad=True, device=device)
    log_var_bc_r_max = torch.zeros((1,), requires_grad=True, device=device)
    log_var_bc_theta_0 = torch.zeros((1,), requires_grad=True, device=device)
    log_var_bc_theta_pi = torch.zeros((1,), requires_grad=True, device=device)

    optimizer = optim.Adam(
        list(model.parameters()) + log_vars + [log_var_ic, log_var_bc_r_min,
                                               log_var_bc_r_max, log_var_bc_theta_0, log_var_bc_theta_pi],
        lr=0.0005, weight_decay=1e-5
    )

    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_val_loss = float('inf')

    history = {
        'train_loss': [], 'val_loss': [], 'mse_loss': [],
        'cont_loss': [], 'mom_r_loss': [], 'mom_theta_loss': [], 'energy_loss': [],
        'ic_loss': [], 'bc_r_min_loss': [],
        'bc_r_max_loss': [], 'bc_theta_0_loss': [], 'bc_theta_pi_loss': []
    }

    R_std, Theta_std, T_std = standard_values_X
    R_min_norm = X_train[:, 0].min().item()
    R_max_norm = X_train[:, 0].max().item()
    theta_min_norm = X_train[:, 1].min().item()
    theta_max_norm = X_train[:, 1].max().item()

    # 记录总训练开始时间
    global_start_time = time.time()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, total_mse, total_cont, total_mom_r, total_mom_theta, total_energy = [0.0] * 6
        total_ic, total_bc_r_min, total_bc_r_max, total_bc_theta_0, total_bc_theta_pi = 0.0, 0.0, 0.0, 0.0, 0.0
        count_ic, count_bc_r_min, count_bc_r_max, count_bc_theta_0, count_bc_theta_pi = 0, 0, 0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad_(True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                mse_loss_val = criterion(outputs, targets)
                physics_losses = compute_physics_loss(outputs, inputs, standard_values_X, standard_values_Y)

                # 初始条件损失
                t_norm = inputs[:, 2]
                ic_mask = (t_norm < 1e-5)
                ic_loss = torch.tensor(0.0, device=device)
                if ic_mask.any():
                    ic_loss = criterion(outputs[ic_mask], targets[ic_mask])
                    count_ic += ic_mask.sum().item()

                # r_min 边界: Vr = 0
                r_norm = inputs[:, 0]
                bc_r_min_mask = (r_norm < R_min_norm + 1e-5)
                bc_r_min_loss = torch.tensor(0.0, device=device)
                if bc_r_min_mask.any():
                    Vr_pred = outputs[bc_r_min_mask, 1]
                    bc_r_min_loss = torch.mean(Vr_pred ** 2)
                    count_bc_r_min += bc_r_min_mask.sum().item()

                # r_max 边界: ∂rho/∂r = 0
                bc_r_max_mask = (r_norm > R_max_norm - 1e-5)
                bc_r_max_loss = torch.tensor(0.0, device=device)
                if bc_r_max_mask.any():
                    r_max_points = inputs[bc_r_max_mask]
                    r_max_points.requires_grad_(True)
                    outputs_r_max = model(r_max_points)
                    rho_r_max = outputs_r_max[:, 4]

                    drho_dr = torch.autograd.grad(
                        outputs=rho_r_max,
                        inputs=r_max_points,
                        grad_outputs=torch.ones_like(rho_r_max),
                        create_graph=True,
                        retain_graph=True
                    )[0][:, 0]

                    bc_r_max_loss = torch.mean(drho_dr ** 2)
                    count_bc_r_max += bc_r_max_mask.sum().item()

                # θ = 0 边界
                theta_norm = inputs[:, 1]
                bc_theta_0_mask = (theta_norm < theta_min_norm + 1e-5)
                bc_theta_0_loss = torch.tensor(0.0, device=device)
                if bc_theta_0_mask.any():
                    Vtheta_theta_0 = outputs[bc_theta_0_mask, 2]
                    loss_Vtheta = torch.mean(Vtheta_theta_0 ** 2)

                    theta_0_points = inputs[bc_theta_0_mask]
                    theta_0_points.requires_grad_(True)
                    outputs_theta_0 = model(theta_0_points)

                    scalar_indices = [4, 0, 5, 3, 6]
                    dscalar_dtheta = []
                    for idx in scalar_indices:
                        field = outputs_theta_0[:, idx]
                        dfield_dtheta = torch.autograd.grad(
                            outputs=field,
                            inputs=theta_0_points,
                            grad_outputs=torch.ones_like(field),
                            create_graph=True,
                            retain_graph=True
                        )[0][:, 1]
                        dscalar_dtheta.append(dfield_dtheta)

                    loss_scalar = torch.mean(torch.stack([d ** 2 for d in dscalar_dtheta]))
                    bc_theta_0_loss = loss_Vtheta + loss_scalar
                    count_bc_theta_0 += bc_theta_0_mask.sum().item()

                # θ = π 边界
                bc_theta_pi_mask = (theta_norm > theta_max_norm - 1e-5)
                bc_theta_pi_loss = torch.tensor(0.0, device=device)
                if bc_theta_pi_mask.any():
                    Vtheta_theta_pi = outputs[bc_theta_pi_mask, 2]
                    loss_Vtheta = torch.mean(Vtheta_theta_pi ** 2)

                    theta_pi_points = inputs[bc_theta_pi_mask]
                    theta_pi_points.requires_grad_(True)
                    outputs_theta_pi = model(theta_pi_points)

                    scalar_indices = [4, 0, 5, 3, 6]
                    dscalar_dtheta = []
                    for idx in scalar_indices:
                        field = outputs_theta_pi[:, idx]
                        dfield_dtheta = torch.autograd.grad(
                            outputs=field,
                            inputs=theta_pi_points,
                            grad_outputs=torch.ones_like(field),
                            create_graph=True,
                            retain_graph=True
                        )[0][:, 1]
                        dscalar_dtheta.append(dfield_dtheta)

                    loss_scalar = torch.mean(torch.stack([d ** 2 for d in dscalar_dtheta]))
                    bc_theta_pi_loss = loss_Vtheta + loss_scalar
                    count_bc_theta_pi += bc_theta_pi_mask.sum().item()

                loss_mse = torch.exp(-log_vars[0]) * mse_loss_val + 0.5 * log_vars[0]
                loss_cont = torch.exp(-log_vars[1]) * physics_losses['cont'] + 0.5 * log_vars[1]
                loss_mom_r = torch.exp(-log_vars[2]) * physics_losses['mom_r'] + 0.5 * log_vars[2]
                loss_mom_theta = torch.exp(-log_vars[3]) * physics_losses['mom_theta'] + 0.5 * log_vars[3]
                loss_energy = torch.exp(-log_vars[4]) * physics_losses['energy'] + 0.5 * log_vars[4]
                loss_ic = torch.exp(-log_var_ic) * ic_loss + 0.5 * log_var_ic
                loss_bc_r_min = torch.exp(-log_var_bc_r_min) * bc_r_min_loss + 0.5 * log_var_bc_r_min
                loss_bc_r_max = torch.exp(-log_var_bc_r_max) * bc_r_max_loss + 0.5 * log_var_bc_r_max
                loss_bc_theta_0 = torch.exp(-log_var_bc_theta_0) * bc_theta_0_loss + 0.5 * log_var_bc_theta_0
                loss_bc_theta_pi = torch.exp(-log_var_bc_theta_pi) * bc_theta_pi_loss + 0.5 * log_var_bc_theta_pi

                loss = (loss_mse + loss_cont + loss_mom_r + loss_mom_theta + loss_energy +
                        loss_bc_r_min + loss_bc_r_max +
                        loss_bc_theta_0 + loss_bc_theta_pi)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            total_mse += mse_loss_val.item() * inputs.size(0)
            total_cont += physics_losses['cont'].item() * inputs.size(0)
            total_mom_r += physics_losses['mom_r'].item() * inputs.size(0)
            total_mom_theta += physics_losses['mom_theta'].item() * inputs.size(0)
            total_energy += physics_losses['energy'].item() * inputs.size(0)
            total_ic += ic_loss.item() * (ic_mask.sum().item() if ic_mask.any() else 0)
            total_bc_r_min += bc_r_min_loss.item() * (bc_r_min_mask.sum().item() if bc_r_min_mask.any() else 0)
            total_bc_r_max += bc_r_max_loss.item() * (bc_r_max_mask.sum().item() if bc_r_max_mask.any() else 0)
            total_bc_theta_0 += bc_theta_0_loss.item() * (bc_theta_0_mask.sum().item() if bc_theta_0_mask.any() else 0)
            total_bc_theta_pi += bc_theta_pi_loss.item() * (
                bc_theta_pi_mask.sum().item() if bc_theta_pi_mask.any() else 0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_mse = total_mse / len(train_loader.dataset)
        avg_cont = total_cont / len(train_loader.dataset)
        avg_mom_r = total_mom_r / len(train_loader.dataset)
        avg_mom_theta = total_mom_theta / len(train_loader.dataset)
        avg_energy = total_energy / len(train_loader.dataset)
        avg_ic = total_ic / max(count_ic, 1)
        avg_bc_r_min = total_bc_r_min / max(count_bc_r_min, 1)
        avg_bc_r_max = total_bc_r_max / max(count_bc_r_max, 1)
        avg_bc_theta_0 = total_bc_theta_0 / max(count_bc_theta_0, 1)
        avg_bc_theta_pi = total_bc_theta_pi / max(count_bc_theta_pi, 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(device))
                val_loss += criterion(outputs, targets.to(device)).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['mse_loss'].append(avg_mse)
        history['cont_loss'].append(avg_cont)
        history['mom_r_loss'].append(avg_mom_r)
        history['mom_theta_loss'].append(avg_mom_theta)
        history['energy_loss'].append(avg_energy)
        history['ic_loss'].append(avg_ic)
        history['bc_r_min_loss'].append(avg_bc_r_min)
        history['bc_r_max_loss'].append(avg_bc_r_max)
        history['bc_theta_0_loss'].append(avg_bc_theta_0)
        history['bc_theta_pi_loss'].append(avg_bc_theta_pi)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pinn_mlp_model.pth')

        print(
            f'Epoch {epoch + 1}/{epochs} | Time: {time.time() - start_time:.2f}s | Train Loss: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}')
        print(
            f'    Losses -> MSE: {avg_mse:.6f} | Cont: {avg_cont:.6f} | Mom_r: {avg_mom_r:.6f} | Mom_t: {avg_mom_theta:.6f} | Energy: {avg_energy:.6f}')
        print(
            f'    BC Losses -> IC: {avg_ic:.6f} | r_min: {avg_bc_r_min:.6f} | r_max: {avg_bc_r_max:.6f} | θ=0: {avg_bc_theta_0:.6f} | θ=π: {avg_bc_theta_pi:.6f}')

    # 打印总训练时长
    global_end_time = time.time()
    total_time_seconds = global_end_time - global_start_time
    m, s = divmod(total_time_seconds, 60)
    h, m = divmod(m, 60)

    print("\n" + "=" * 60)
    print(f"Training Completed!")
    print(f"Total Training Time: {int(h)}h {int(m)}m {s:.2f}s")
    print(f"Best Val MSE = {best_val_loss:.6f} -> Saved to 'best_pinn_mlp_model.pth'")
    print("=" * 60 + "\n")

    return model, history


# 6. 绘图与评估函数
def plot_loss_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['mse_loss'], 'b-o', label='Training MSE Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    ax1.set_title('MSE and Validation Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, ls="--")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['cont_loss'], 'g-^', label='Continuity')
    ax2.plot(epochs, history['mom_r_loss'], 'c-^', label='Momentum-r')
    ax2.plot(epochs, history['mom_theta_loss'], 'm-^', label='Momentum-θ')
    ax2.plot(epochs, history['energy_loss'], 'y-^', label='Energy')
    ax2.set_title('Physics Losses')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, ls="--")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['bc_r_min_loss'], 's-', color='orange', label='BC (r_min: Vr=0)')
    ax3.plot(epochs, history['bc_r_max_loss'], 's-', color='purple', label='BC (r_max: ∂rho/∂r=0)')
    ax3.plot(epochs, history['bc_theta_0_loss'], 's-', color='green', label='BC (θ=0)')
    ax3.plot(epochs, history['bc_theta_pi_loss'], 's-', color='blue', label='BC (θ=π)')
    ax3.set_title('Boundary Conditions')
    ax3.set_yscale('log')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, ls="--")

    plt.tight_layout()
    plt.savefig("loss_curves_with_full_bc.png", dpi=300)
    plt.show()


# ============================================================
# 推理时间评估模块
# ============================================================
def measure_inference_time(model, X_sample, device, num_runs=100):
    model.eval()
    X_sample = X_sample.to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(X_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(X_sample)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_runs) * 1000

    print(f"  - Batch Size: {X_sample.shape[0]}")
    print(f"  - Total time for {num_runs} runs: {total_time:.4f} s")
    print(f"  - Average time per forward pass:  {avg_time_ms:.4f} ms")


def calculate_final_error(model, X_val, Y_val, standard_values_Y, physical_names):
    device = next(model.parameters()).device
    model.eval()

    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    standard_values_Y_tensor = torch.tensor(standard_values_Y, dtype=torch.float32, device=device)

    with torch.no_grad():
        Y_pred_scaled = model(X_val)
        Y_pred = Y_pred_scaled
        Y_true = Y_val

        print("\n--- Final Relative L2 Error on Validation Set ---")
        for i, name in enumerate(physical_names):
            true_quantity = Y_true[:, i]
            pred_quantity = Y_pred[:, i]
            diff_norm = torch.linalg.norm(true_quantity - pred_quantity)
            true_norm = torch.linalg.norm(true_quantity)
            if true_norm == 0:
                relative_error = 0.0 if diff_norm == 0 else float('inf')
            else:
                relative_error = diff_norm / true_norm
            print(f"  - {name}: {relative_error.item() * 100:.4f}%")
        print("-------------------------------------------------")


def visualize_cross_section(r, theta, actual, predicted, title, time_step, angle_deg=90):
    target_angle_rad = np.deg2rad(angle_deg)
    angle_tolerance = np.deg2rad(1.0)
    indices = np.where(np.abs(theta - target_angle_rad) < angle_tolerance)

    if len(indices[0]) == 0:
        print(f"Warning: No data points found near {angle_deg} degrees.")
        return None

    r_slice = r[indices]
    actual_slice = actual[indices]
    predicted_slice = predicted[indices]

    sort_order = np.argsort(r_slice)
    r_slice = r_slice[sort_order]
    actual_slice = actual_slice[sort_order]
    predicted_slice = predicted_slice[sort_order]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f'Cross-Section of {title} at {angle_deg}° (t={time_step:.4f})', fontsize=16)

    ax1.plot(r_slice, actual_slice, 'b-', label='Actual', linewidth=2)
    ax1.plot(r_slice, predicted_slice, 'r--', label='Predicted', linewidth=2)
    ax1.set_ylabel(title)
    ax1.legend()
    ax1.grid(True, linestyle=':')
    ax1.set_title('Shock Front Profile Comparison')

    error_slice = actual_slice - predicted_slice
    ax2.plot(r_slice, error_slice, 'k-', label='Error (Actual - Predicted)')
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel('Radius (r)')
    ax2.set_ylabel('Error')
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f'pinn_mlp_{title}_cross_section_{angle_deg}deg_t{time_step:.4f}.png'
    plt.savefig(save_path, dpi=300)
    print(f"    Saved cross-section figure to {save_path}")
    plt.show()
    return fig


def evaluate_and_visualize(model, standard_values_X, standard_values_Y, sampled_vertices_indices, sampled_times,
                           coordinates_filtered,
                           E_full, Vr_full, Vtheta_full, Qr_full, rho_full, P_full, Qtheta_full,
                           visualization_subsample=0.2):
    device = next(model.parameters()).device
    model.eval()
    print("Starting evaluation for visualization...")
    physical_names = ['E', 'Vr', 'Vtheta', 'Qr', 'rho', 'P', 'Qtheta']
    full_data_arrays = [E_full, Vr_full, Vtheta_full, Qr_full, rho_full, P_full, Qtheta_full]

    plot_times = np.random.choice(sampled_times, min(3, len(sampled_times)), replace=False)

    for time_step in plot_times:
        print(f"\n--- Preparing data for visualization at time_step = {time_step} ---")

        r_coords = coordinates_filtered[:, 0]
        theta_coords = coordinates_filtered[:, 1]
        num_spatial_points = len(r_coords)

        X_time = np.zeros((num_spatial_points, 3), dtype=np.float32)
        X_time[:, 0] = r_coords
        X_time[:, 1] = theta_coords
        X_time[:, 2] = time_step

        X_time_scaled = X_time / standard_values_X

        print(f"Predicting {num_spatial_points} points for time_step {time_step}...")
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_time_scaled, dtype=torch.float32).to(device)).cpu().numpy()
            preds = preds_scaled * standard_values_Y

        time_idx_in_original_times = np.where(np.linspace(0, 1, E_full.shape[1]) == time_step)[0][0]

        actual_data = {
            name: data[sampled_vertices_indices, time_idx_in_original_times] for name, data in
            zip(physical_names, full_data_arrays)
        }

        if visualization_subsample < 1.0:
            num_plot_points = int(num_spatial_points * visualization_subsample)
            subsample_indices = np.random.choice(num_spatial_points, num_plot_points, replace=False)
            plot_r = coordinates_filtered[subsample_indices, 0]
            plot_theta = coordinates_filtered[subsample_indices, 1]
            plot_preds = preds[subsample_indices, :]
            plot_actuals = {name: data[subsample_indices] for name, data in actual_data.items()}
        else:
            plot_r, plot_theta = coordinates_filtered[:, 0], coordinates_filtered[:, 1]
            plot_preds = preds
            plot_actuals = actual_data

        for i, name in enumerate(physical_names):
            print(f"  - Visualizing '{name}'...")
            actual_vals = plot_actuals[name]
            predicted_vals = plot_preds[:, i]

            try:
                fig = visualize_comparison(plot_r, plot_theta, actual_vals, predicted_vals, name, time_step)
                save_path = f'pinn_mlp_{name}_comparison_t{time_step:.4f}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved figure to {save_path}")

                # 提取 rho 的截面图分析
                if name == 'rho':
                    visualize_cross_section(
                        plot_r, plot_theta, actual_vals, predicted_vals,
                        name, time_step, angle_deg=30
                    )

            except Exception as e:
                print(f"    ERROR: Could not visualize {name} at t={time_step}: {e}")
    print("\nVisualization complete.")


def visualize_comparison(r, theta, actual, predicted, title, time_step, figsize=(18, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw=dict(projection='polar'))
    vmin = min(np.nanmin(actual), np.nanmin(predicted)) if not (
            np.all(np.isnan(actual)) or np.all(np.isnan(predicted))) else 0
    vmax = max(np.nanmax(actual), np.nanmax(predicted)) if not (
            np.all(np.isnan(actual)) or np.all(np.isnan(predicted))) else 1
    common_params = {'cmap': 'viridis', 's': 10, 'alpha': 0.8, 'vmin': vmin, 'vmax': vmax}

    sc1 = axes[0].scatter(theta, r, c=actual, **common_params)
    axes[0].set_title(f'Actual {title} at t={time_step:.4f}', pad=20)
    fig.colorbar(sc1, ax=axes[0])
    sc2 = axes[1].scatter(theta, r, c=predicted, **common_params)
    axes[1].set_title(f'Predicted {title} at t={time_step:.4f}', pad=20)
    fig.colorbar(sc2, ax=axes[1])
    sc3 = axes[2].scatter(theta, r, c=np.abs(actual - predicted), cmap='hot', s=10, alpha=0.8)
    axes[2].set_title('Absolute Error', pad=20)
    fig.colorbar(sc3, ax=axes[2])
    plt.tight_layout()
    return fig


# 7. 主程序
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 控制开关,MODEL_MODE='positional_encoding'or'basic'
    TRAIN_NEW_MODEL = True
    MODEL_PATH = 'best_pinn_mlp_model.pth'
    MODEL_MODE = 'positional_encoding'

    print("Loading data...")
    r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta = load_data()

    print("Preprocessing data...")
    X_train, Y_train, X_val, Y_val, standard_values_X, standard_values_Y, s_v, s_t, coords = preprocess_data(
        r, theta, E, Vr, Vtheta, Qr, rho, P, Qtheta,
        spatial_stride=1, time_stride=1, r_threshold=0.045
    )

    input_dim = 3
    model = PinnMLP(
        input_size=input_dim,
        hidden_size=256,
        output_size=7,
        num_hidden_layers=4,
        mode=MODEL_MODE,
        num_freqs=10,
    )
    print(f"Successfully created model with mode: '{MODEL_MODE}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if TRAIN_NEW_MODEL:
        print("Initializing weights for training...")
        model.apply(init_weights)
        Y_mean = Y_train.mean(dim=0)
        with torch.no_grad():
            model.network[-1].bias.copy_(Y_mean)

        print("Starting model training...")
        trained_model, history = train_model(
            model,
            X_train, Y_train, X_val, Y_val,
            standard_values_X, standard_values_Y,
            epochs=100,
            batch_size=4096
        )
        print("Plotting loss curves...")
        plot_loss_curves(history)
    else:
        print(f"Loading pre-trained model from '{MODEL_PATH}'...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Please check the path or set TRAIN_NEW_MODEL to True.")

        start_time = time.perf_counter()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        end_time = time.perf_counter()
        loading_time = end_time - start_time
        print(f"Model weights loaded successfully in {loading_time:.4f} seconds.")

        trained_model = model

        print("\n--- Inference Time Test ---")
        single_input = X_val[0:1]
        print("Test 1: Single Sample (Latency)")
        measure_inference_time(trained_model, single_input, device, num_runs=1000)

        print(f"\nTest 2: Full Validation Set ({len(X_val)} samples)")
        measure_inference_time(trained_model, X_val, device, num_runs=100)
        print("---------------------------\n")

    print("\nSetting model to evaluation mode for inference...")
    trained_model.eval()

    print("Calculating final error on validation set...")
    physical_names = ['E', 'Vr', 'Vtheta', 'Qr', 'rho', 'P', 'Qtheta']
    calculate_final_error(
        trained_model,
        X_val,
        Y_val,
        standard_values_Y,
        physical_names
    )

    print("Generating visualizations...")
    evaluate_and_visualize(
        trained_model, standard_values_X, standard_values_Y, s_v, s_t, coords,
        E, Vr, Vtheta, Qr, rho, P, Qtheta,
        visualization_subsample=1.0
    )
    print("\nProcess finished.")


if __name__ == "__main__":
    main()
