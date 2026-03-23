import numpy as np
import pandas as pd
import os
import random
import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
import glob

tc.set_num_threads(14)

#--------------------------------------------------------------------------------------------------

num_replicas = 200    # Number of MC Replicas
n_epochs = 1000000     # Number of epochs
target_loss = 5e1     # Stop training, it's a great loss.

patience_limit = 10000  # Wait that long; if the learning stagnates, return to the best value of the loss (jump).
threshold = 7.0e1       # It only saves if the loss (cost function) is less than that.
max_jumps = 3           # Limit of times the network can "jump" before Early Stopping.

#--------------------------------------------------------------------------------------------------

# =============================================================================
# 1. SEEDS FOR REPRODUCIBILITY
# =============================================================================
def fix_seeds(seed=None):
    """
    Ensures deterministic behavior by fixing seeds across all libraries.
    """
    if seed is None:
        seed = random.randrange(0, 2**32)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    
    # If using GPU, ensure deterministic CUDA kernels
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)
        tc.backends.cudnn.deterministic = True
        tc.backends.cudnn.benchmark = False
        
    print(f"SEED = {seed}")
    return seed

seed_choice = fix_seeds(1276791801)

#--------------------------------------------------------------------------------------------------

# =============================================================================
# 2. DATA STRUCTURES & LOADING
# =============================================================================

@dataclass
class NormalizationStats:
    """Stores normalization constants for post-training inference and plotting."""
    log_s_mean: tc.Tensor
    log_s_std : tc.Tensor
    Delta_mean: tc.Tensor
    Delta_std : tc.Tensor
    Y_mean    : tc.Tensor
    Y_std     : tc.Tensor

class ScatteringData:
    def __init__(self, csv_path="data.csv", device="cpu"):
        self.device = device
        self._load_and_process(csv_path)

    def _load_and_process(self, csv_path):
        # --- Loading and Initial Cleaning ---
        df = pd.read_csv(csv_path)
        # Drop NaN values in critical physics columns
        cols_req = ["sqrt_s_GeV", "t_GeV2", "dsig_dt_mb_GeV2"]
        df_fit = df.dropna(subset=cols_req).copy()
        # Filter out unphysical or zero cross-sections
        df_fit = df_fit[df_fit["dsig_dt_mb_GeV2"] > 1e-15]
        self.df_fit = df_fit

        # --- Physics Variable Transformation ---
        # Convert to float32 early to save memory and ensure torch compatibility
        dsig_dt = df_fit["dsig_dt_mb_GeV2"].values.astype(np.float32)
        sqrt_s  = df_fit["sqrt_s_GeV"].values.astype(np.float32)
        t_abs   = np.abs(df_fit["t_GeV2"].values.astype(np.float32))

        # Log-scale is preferred for cross-sections due to high dynamic range
        s_pp = sqrt_s **2
        log_s = np.log(s_pp).astype(np.float32)
        Delta = np.sqrt(t_abs)
        c_star = 1e0 # GeV
        Y     = np.log(dsig_dt/c_star)

        # Absolute Error for generate replicas    
        Err_plus = df_fit["err_total_plus"].values.astype(np.float32)
        Err_mnus = df_fit["err_total_minus"].values.astype(np.float32)                

        # Relative errors (normalized by the central value for log-space training)
        # Errors normalized by Y_std to remain consistent with Y_tc scale
        Err_log_p = Err_plus / dsig_dt
        Err_log_m = Err_mnus / dsig_dt

        # --- Tensor Conversion for torch & Device Placement ---
        # Helper to convert numpy arrays to PyTorch tensors efficiently
        def prep(arr): 
            return tc.from_numpy(arr).view(-1, 1).float().to(self.device)
        
        self.Y_tc         = prep(Y)
        self.Delta_tc     = prep(Delta)
        self.sqrt_s_tc = prep(sqrt_s)
        self.s_tc         = prep(s_pp)
        self.log_s_tc = prep(log_s)

        # Absolute Error for generate replicas    
        self.Err_plus_tc  = prep(Err_plus)
        self.Err_mnus_tc  = prep(Err_mnus)

        self.Err_log_p_tc = prep(Err_log_p)
        self.Err_log_m_tc = prep(Err_log_m)

        # --- Constants for Z-score (Calculated ONCE) ---
        # We store everything in the stats object first
        self.stats = NormalizationStats(
            log_s_mean = self.log_s_tc.mean(),
            log_s_std  = self.log_s_tc.std() + 1e-12,
            Delta_mean = self.Delta_tc.mean(),
            Delta_std  = self.Delta_tc.std() + 1e-12,
            Y_mean     = self.Y_tc.mean(),
            Y_std      = self.Y_tc.std() + 1e-12
        )
        
        # --- Shortcut attributes (Pointing to the same memory) ---
        # This satisfies your _normalize method without recalculating anything
        self.log_s_tc_mean = self.stats.log_s_mean
        self.log_s_tc_std  = self.stats.log_s_std
        self.Delta_tc_mean = self.stats.Delta_mean
        self.Delta_tc_std  = self.stats.Delta_std
        
    # =============================================================================
    # REPLICAS & SPLITTING
    # =============================================================================    

    def _subset(self, indices):
        """
        Creates a lightweight copy of the dataset for a specific subset of indices.
        """
        # Create a new instance without calling __init__ to avoid reloading CSV
        subset = self.__class__.__new__(self.__class__)
        subset.device = self.device
        subset.stats  = self.stats # Reference same stats

        subset.log_s_tc_mean = self.log_s_tc_mean
        subset.log_s_tc_std  = self.log_s_tc_std
        subset.Delta_tc_mean = self.Delta_tc_mean
        subset.Delta_tc_std  = self.Delta_tc_std
        
        # Slice only the necessary tensors
        subset.s_tc     = self.s_tc[indices]
        subset.log_s_tc = self.log_s_tc[indices]
        subset.sqrt_s_tc = self.sqrt_s_tc[indices]
        subset.Delta_tc     = self.Delta_tc[indices]
        subset.Y_tc         = self.Y_tc[indices]
        subset.Err_plus_tc  = self.Err_plus_tc[indices]
        subset.Err_mnus_tc  = self.Err_mnus_tc[indices]
        subset.Err_log_p_tc = self.Err_log_p_tc[indices]
        subset.Err_log_m_tc = self.Err_log_m_tc[indices]
        
        # Keep track of original energy for plotting subsets
        if hasattr(self, 'sqrt_s_tc'):
            sqrt_s_tc = tc.sqrt(self.s_tc)
            subset.sqrt_s_tc = sqrt_s_tc[indices]
            
        return subset

    def generate_replica_split(self, split_ratio=0.8):
        """
        Generates a Monte Carlo replica by fluctuating data points within 
        experimental errors and splitting by energy levels (sqrt_s).
        """
        # 1. Generate Gaussian fluctuations (Asymmetric)
        z_random = tc.randn_like(self.Y_tc)
        # Apply plus or mnus error based on the sign of the random shift
        fluctuation = tc.where(z_random > 0, 
                               z_random * self.Err_log_p_tc, 
                               z_random * self.Err_log_m_tc)
        
        # Create a temporary clone for the full replica values
        y_replica_values = self.Y_tc.clone() + fluctuation

        # 2. Energy-based splitting (Golden Rule for Scattering Data)
        sqrt_s_flat = self.sqrt_s_tc.view(-1)
        unique_energy = tc.unique(sqrt_s_flat)
        
        n_unique = unique_energy.numel()
        shuffled_idx = tc.randperm(n_unique, device=self.device)

        split_mark = int(n_unique * split_ratio)
        train_energies = unique_energy[shuffled_idx[:split_mark]]
        
        # Create masks
        # Logical 'isin' checks which points belong to the selected training energies
        train_mask = tc.isin(sqrt_s_flat, train_energies)
        
        # Indices extraction
        train_idx = tc.where(train_mask)[0]
        val_idx   = tc.where(~train_mask)[0] # Improved: val is simply NOT train

        # 3. Build the subsets
        data_train = self._subset(train_idx)
        data_val   = self._subset(val_idx)
        
        # Apply the fluctuated Y values to the training set only (or both, depending on your methodology)
        # Usually, validation uses original data, but for replicas, we use fluctuated for both.
        data_train.Y_tc = y_replica_values[train_idx]
        data_val.Y_tc   = y_replica_values[val_idx]

        return data_train, data_val

#--------------------------------------------------------------------------------------------------    

#Loads data globally
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

# Instantiate the class by loading the CSV.
data = ScatteringData(csv_path="data.csv", device=device)

# =============================================================================
# 3. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class NNAmplitude(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------- NN residual ----------------
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, 2)
        )
        
        # ------ Initiate the weights and bias -----------
        # We start with zero weights and biases in the last layer to have the theory
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)
    
    # ----------------------------------------------------------------
    # ----------------------- NN function ----------------------------
    # ----------------------------------------------------------------
    def forward(self, log_s_z, Delta_z):
        if log_s_z.dim() == 1:
            log_s_z = log_s_z.view(-1, 1)
        if Delta_z.dim() == 1:
            Delta_z = Delta_z.view(-1, 1)
        # Purely scaled inputs and outputs (Z-score)
        nn_in = tc.cat([log_s_z, Delta_z], dim=1)
        # Return [out_R, out_I]
        return self.net(nn_in) 
    
# =============================================================================
# 4. PHYSICAL MODEL
# =============================================================================

class PhysicalAmplitude(nn.Module): # <--- FIX: Removed ScatteringData from here
    def __init__(self, scattering_instance): # <--- Pass the already loaded data instance
        super().__init__() # Now this correctly initializes ONLY nn.Module

        # Store the reference to the data object
        self.data_ref = scattering_instance
        self.device = scattering_instance.device

        self.register_buffer("log_s_mean", scattering_instance.log_s_tc_mean)
        self.register_buffer("log_s_std",  scattering_instance.log_s_tc_std)
        self.register_buffer("Delta_mean", scattering_instance.Delta_tc_mean)
        self.register_buffer("Delta_std",  scattering_instance.Delta_tc_std)

        # Instantiate the neural network internally
        self.nn_amp = NNAmplitude().to(self.device)            
        
        # ---------------- t-shape baseline ----------------
        # B(s) = B0 + 2 alpha_prime ln(s/s_regge)
        self.register_buffer("B0",          tc.tensor(4.0))  # GeV^-2
        self.register_buffer("alpha_prime", tc.tensor(0.25)) # GeV^-2
        self.register_buffer("log_s_regge", tc.tensor(0.0))  # ln((1 GeV)^2)
                
        # ---------------- Power for dipole-like baseline ----------------
        # n(s) = n0 + n1 * (s/s0)^(-eps)
        self.n0 = nn.Parameter(tc.tensor([4.0]))   # The low-energy pQCD limit (Dimensionless)
        self.n0.requires_grad_(False)
        
        self.n1 = nn.Parameter(tc.tensor([1.0]))   # Growth rate with energy (Dimensionless)
        self.n1.requires_grad_(True)           

        self.epsilon = nn.Parameter(tc.tensor([0.05])) 
        self.epsilon.requires_grad_(True)             

        self.register_buffer("log_s0", tc.tensor(0.0))  # ln((1 GeV)^2)

        # ----------------------------------------------------------------
        # ---------------- sigma_tot: COMPETE-like------------------------
        # ----------------------------------------------------------------
        # Phys. Rev. D 65, 074024 (2002) 
        # C. Amsler et al., Particle Data Group. Phys. Lett. B 667, 1 (2008)

        # s0 = 5.38^2 = 28.9444 GeV^2 -> ln(s0) = 3.36538
        # s1 = 1 GeV^2
        self.register_buffer("log_s0_compete", tc.tensor(3.36538)) # ln(1.0 GeV^2) 
        self.register_buffer("log_s1", tc.tensor(0.0))       # ln(1.0 GeV^2)
        
        self.register_buffer("Z",    tc.tensor(35.45))       # mb
        self.register_buffer("B",    tc.tensor(0.308))       # mb 
        self.register_buffer("Yp",   tc.tensor(42.53))       # mb
        self.register_buffer("etap", tc.tensor(0.458))
        self.register_buffer("Ym",   tc.tensor(33.34))       # mb
        self.register_buffer("etam", tc.tensor(0.545))

        # 1 GeV^{-2} = 0.389379 mb
        self.register_buffer("hbarc2", tc.tensor(0.389379))

        self.to(device)

        # For the integration int dt dsigma/dt 
        # 2. Create the t-grid (momentum transfer)
        t_grid = tc.logspace(np.log10(1e-5), np.log10(50.0), 200)
        self.register_buffer("t_grid_integration", t_grid)

    def _normalize(self, s, Delta):
        """
        Internal method: Now uses INTERNAL BUFFERS for normalization.
        No external data dependency!
        """
        if s.dim() == 1: s = s.view(-1, 1)
        if Delta.dim() == 1: Delta = Delta.view(-1, 1)
        
        ln_s = tc.log(s)
        
        ln_s_z = (ln_s - self.log_s_mean) / self.log_s_std
        Delta_z = (Delta - self.Delta_mean) / self.Delta_std
        
        return ln_s_z, Delta_z

    def sigma_tot(self, s, mode="pp"):
        log_s = tc.log(s)        
        diff_s0 = log_s - self.log_s0_compete # ln(s/s0)
        diff_s1 = log_s - self.log_s1         # ln(s/s1)

        # Regge terms using log-space for stability
        regge_even = self.Yp * tc.exp(-self.etap * diff_s1)
        regge_odd  = self.Ym * tc.exp(-self.etam * diff_s1)
        
        # Pomeron terms (Universal rise)
        pomeron = self.Z + self.B * (diff_s0**2)
        
        even = pomeron + regge_even
        
        # pp is (Even - Odd), pbarp is (Even + Odd)
        if mode == "pp":
            sig = even - regge_odd
        else:
            sig = even + regge_odd
            
        return tc.clamp(sig, min=1e-6)

    # ---------------- rho(s) ----------------------------------------
    # rho(s) via Derivative Dispersion Relation (DDR):
    # rho approx (pi/2) * (1/sigma_tot) * d sigma_tot / d ln(s)
    # ----------------------------------------------------------------
    def rho(self, s, mode="pp"):
        ln_s = tc.log(s)
        # Scales
        L_s0 = ln_s - self.log_s0_compete
        diff_s1 = ln_s - self.log_s1
            
        # Terms (Regge terms in log-space for stability)
        regge_p = self.Yp * tc.exp(-self.etap * diff_s1)
        regge_m = self.Ym * tc.exp(-self.etam * diff_s1)
        pomeron = self.Z + self.B * (L_s0**2)
            
        # Derivatives (d_sigma / d_lns)
        d_pomeron = 2.0 * self.B * L_s0
        d_regge_p = -self.etap * regge_p
        d_regge_m = -self.etam * regge_m
            
        # Signs based on mode
        # pp: even - odd | pbarp: even + odd
        is_pp = 1.0 if mode == "pp" else -1.0
            
        sig = (pomeron + regge_p) - (is_pp * regge_m)
        d_sig = (d_pomeron + d_regge_p) - (is_pp * d_regge_m)
            
        # DDR Formula
        # We use a small epsilon for numerical safety
        rho_val = (tc.pi / 2.0) * (d_sig / (sig + 1e-9))
            
        return tc.clamp(rho_val, min=-0.5, max=0.5)
    
    # ----------------------------------------------------------------
    # ---------------- Elastic Amplitude -----------------------------
    # ----------------- A_el(s, Delta) -------------------------------
    # ----------------------------------------------------------------

    def A_el(self, s, Delta, mode="pp"):
        """
        Calculates the Real and Imaginary components of the amplitude.
        This is the "brain" of the model.
        """
        # A. Physical Scales Conversion
        ln_s_z, Delta_z = self._normalize(s, Delta)
        ln_s = tc.log(s)
        
        # B. Neural Network Residuals
        nn_out = self.nn_amp(ln_s_z, Delta_z)
        out_R, out_I = nn_out[:, 0:1], nn_out[:, 1:2]

        # C. Physics Baseline (COMPETE + Dipole)
        sig_tot = self.sigma_tot(s, mode=mode)
        rho_val = self.rho(s, mode=mode)
        
        n_eff = tc.clamp(self.n0 + self.n1 * tc.exp(-tc.abs(self.epsilon) * (ln_s - self.log_s0)), min=1e-3)
        B_s   = self.B0 + 2.0 * self.alpha_prime * (ln_s - self.log_s_regge)
        
        Aux_dip = tc.clamp(1.0 + (B_s * Delta**2) / n_eff, min=1e-7)
        F_dip   = tc.exp(-n_eff * tc.log(Aux_dip))

        # D. Assembling the Amplitude with NN Corrections
        energy_suppr = tc.sqrt(s)

        f_R = out_R 
        f_I = out_I 

        support = (Delta / energy_suppr)

        # Real (F_R) and Imaginary (F_I) profile components
        F_R =  (rho_val + f_R * support) * F_dip
        F_I = (1.0     + f_I * support) * F_dip

        # Elastic Amplitude components
        Re_A_el = s*sig_tot*F_R
        Im_A_el = s*sig_tot*F_I
        
        return Re_A_el, Im_A_el

    # ----------------------------------------------------------------
    # Differential Cross Section
    # ----------------------------------------------------------------
    def dsigma_dt(self, s, Delta, mode="pp"):
        """
        Orchestrates the amplitude calculation and returns dsigma/dt.
        """
        # Get the components from the amplitude method
        Re_A_el, Im_A_el = self.A_el(s, Delta, mode=mode)

        # Final Calculation: dsigma/dt = (sig_tot^2 / 16*pi*hbarc2) * |F|^2
        mod_A2 = Re_A_el**2 + Im_A_el**2
        dsigma_dt = mod_A2 / (16.0 * tc.pi * self.hbarc2 * s **2)
        
        return dsigma_dt

    # ----------------------------------------------------------------
    # Cross Sections
    # ----------------------------------------------------------------
    def sigmas(self, s, mode="pp"):
        # 1. Ensure s is a 1D vector
        s_flat = s.view(-1)
        num_energies = s_flat.size(0)

        # 2. Access the grid from the BUFFER (instead of creating it here)
        # Use the name you gave in self.register_buffer(...)
        t_grid = self.t_grid_integration 

        # 3. Create the 2D meshgrid
        s_grid_2d, t_grid_2d = tc.meshgrid(s_flat, t_grid, indexing='ij')

        # 4. Flatten for the forward pass
        s_eval = s_grid_2d.reshape(-1, 1)
        Delta_eval = tc.sqrt(t_grid_2d).reshape(-1, 1)

        # 5. Calculate dsigma/dt
        dsig_dt_eval = self.dsigma_dt(s_eval, Delta_eval, mode=mode)
        
        # 6. Reshape back to [Energies, t_points]
        dsig_dt_2d = dsig_dt_eval.view(num_energies, -1)

        # 7. Integration (FIX: Using the buffer t_grid here)
        sig_el = tc.trapezoid(dsig_dt_2d, t_grid, dim=1).view(-1, 1)

        # 8. Analytical sigma_tot
        sigma_tot = self.sigma_tot(s_flat, mode=mode).view(-1, 1)
        
        # 9. Results
        sigma_inel = sigma_tot - sig_el
        ratio = sig_el / (sigma_tot + 1e-12) # Always < 1

        return sig_el, sigma_inel, sigma_tot, ratio
    
#--------------------------------------------------------------------------------------------------    

# =============================================================================
# LOADING AND ENSEMBLE THE REPLICA
# =============================================================================

# Lists all replica models
replica_files = sorted(glob.glob("modelo_replica_*.pth"))

if len(replica_files) == 0:
    raise RuntimeError("Nenhum arquivo 'modelo_replica_*.pth' foi encontrado.")

print(f"Encontradas {len(replica_files)} réplicas:")

replica_models = []
for f in replica_files:
    print("  ", f)
    m = PhysicalAmplitude(data).to(device)

    state_dict = tc.load(f, map_location=device)
    m.load_state_dict(state_dict)

    m.eval()
    replica_models.append(m)

print(f"\nTotal de réplicas carregadas: {len(replica_models)}")


def ensemble_dsdt(replica_models, sqrt_s_val, t_dense, mode="pp"):
    """
    Calcula a média e o desvio padrão do ensemble para dσ/dt
    em uma energia sqrt_s_val e num grid t_dense.
    """

    s_tc = tc.full(
        (len(t_dense), 1),
        sqrt_s_val**2,
        dtype=tc.float32,
        device=device
    )

    t_tc = tc.tensor(t_dense, dtype=tc.float32, device=device).view(-1, 1)
    Delta_tc = tc.sqrt(t_tc)

    preds = []

    with tc.inference_mode():
        for m in replica_models:
            dsdt = m.dsigma_dt(s_tc, Delta_tc, mode=mode)
            preds.append(dsdt.squeeze(-1).cpu().numpy())

    preds = np.array(preds)  # shape = [n_replicas, n_t]

    mean_pred = preds.mean(axis=0)

    if preds.shape[0] > 1:
        std_pred = preds.std(axis=0, ddof=1)
    else:
        std_pred = np.zeros_like(mean_pred)

    return mean_pred, std_pred, preds

class EnsembleMeanModel:
    def __init__(self, replica_models):
        if len(replica_models) == 0:
            raise ValueError("replica_models está vazio.")

        self.replica_models = replica_models
        self.device = replica_models[0].device

    def eval(self):
        for m in self.replica_models:
            m.eval()
        return self

    def _stack_outputs(self, fn_name, *args, **kwargs):
        vals = []
        with tc.inference_mode():
            for m in self.replica_models:
                fn = getattr(m, fn_name)
                vals.append(fn(*args, **kwargs))
        return tc.stack(vals, dim=0)  # [n_rep, ...]

    def dsigma_dt_all(self, s, Delta, mode="pp"):
        return self._stack_outputs("dsigma_dt", s, Delta, mode=mode)

    def dsigma_dt(self, s, Delta, mode="pp"):
        vals = self.dsigma_dt_all(s, Delta, mode=mode)
        return vals.mean(dim=0)

    def dsigma_dt_std(self, s, Delta, mode="pp"):
        vals = self.dsigma_dt_all(s, Delta, mode=mode)
        if vals.size(0) > 1:
            return vals.std(dim=0, unbiased=True)
        return tc.zeros_like(vals[0])

    def sigma_tot_all(self, s, mode="pp"):
        return self._stack_outputs("sigma_tot", s, mode=mode)

    def sigma_tot(self, s, mode="pp"):
        vals = self.sigma_tot_all(s, mode=mode)
        return vals.mean(dim=0)

    def sigma_tot_std(self, s, mode="pp"):
        vals = self.sigma_tot_all(s, mode=mode)
        if vals.size(0) > 1:
            return vals.std(dim=0, unbiased=True)
        return tc.zeros_like(vals[0])

    def rho_all(self, s, mode="pp"):
        return self._stack_outputs("rho", s, mode=mode)

    def rho(self, s, mode="pp"):
        vals = self.rho_all(s, mode=mode)
        return vals.mean(dim=0)

    def rho_std(self, s, mode="pp"):
        vals = self.rho_all(s, mode=mode)
        if vals.size(0) > 1:
            return vals.std(dim=0, unbiased=True)
        return tc.zeros_like(vals[0])


# modelo médio do ensemble
best_model = EnsembleMeanModel(replica_models).eval()

# -------------------------------------------------------------
# EXIBINDO PARÂMETROS FÍSICOS DA MELHOR REDE (Exemplo pegando a última aprovada)
# -------------------------------------------------------------

n0_vals = np.array([m.n0.item() for m in replica_models])
n1_vals = np.array([m.n1.item() for m in replica_models])
ep_vals = np.array([m.epsilon.item() for m in replica_models])

n0_std = n0_vals.std(ddof=1) if len(n0_vals) > 1 else 0.0
n1_std = n1_vals.std(ddof=1) if len(n1_vals) > 1 else 0.0
ep_std = ep_vals.std(ddof=1) if len(ep_vals) > 1 else 0.0

print("\nPARÂMETROS FÍSICOS DO ENSEMBLE:")
print(f"n0 = {n0_vals.mean():.4f} ± {n0_std:.4f}")
print(f"n1 = {n1_vals.mean():.4f} ± {n1_std:.4f}")
print(f"ep = {ep_vals.mean():.4f} ± {ep_std:.4f}")

# =============================================================================
# EXPORTING STANDALONE ENSEMBLE MODEL (No CSV needed for future use)
# =============================================================================

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica carregada para exportar.")

export_stats = {
    'log_s_mean': data.log_s_tc_mean.item(),
    'log_s_std':  data.log_s_tc_std.item(),
    'Delta_mean': data.Delta_tc_mean.item(),
    'Delta_std':  data.Delta_tc_std.item()
}

final_export_dict = {
    'model_class': 'PhysicalAmplitude',
    'n_replicas': len(replica_models),
    'replica_state_dicts': [m.state_dict() for m in replica_models],
    'stats': export_stats
}

tc.save(final_export_dict, "scattering_ensemble_standalone.pth")
print("\n✅ Standalone ensemble salvo como 'scattering_ensemble_standalone.pth'")

# -------------------------------------------------------------
# GRÁFICO DE dσ/dt COM MÉDIA DO ENSEMBLE + BANDA DE INCERTEZA
# -------------------------------------------------------------

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica encontrada: modelo_replica_*.pth")

target_energies = [23.5, 30.7, 44.7, 52.8, 62.5, 7000.0, 13000.0]
t_dense = np.linspace(0.001, 6.0, 500)

fig, ax = plt.subplots(figsize=(10, 10))

for i, E in enumerate(target_energies):
    scale_factor = 10**(-2 * i)

    # ---------------- dados experimentais ----------------
    tol = 0.5 if E < 100 else 50.0
    mask = (data.df_fit['sqrt_s_GeV'] > E - tol) & (data.df_fit['sqrt_s_GeV'] < E + tol)
    df_E = data.df_fit[mask]

    if len(df_E) > 0:
        x_data = np.abs(df_E['t_GeV2'].values)
        y_data = df_E['dsig_dt_mb_GeV2'].values * scale_factor

        err_p = np.abs(df_E['err_total_plus'].values) * scale_factor
        err_m = np.abs(df_E['err_total_minus'].values) * scale_factor

        # segurança para escala log
        err_m = np.minimum(err_m, 0.9999 * y_data)

        ax.errorbar(
            x_data,
            y_data,
            yerr=[err_m, err_p],
            fmt='o',
            color='red',
            markersize=3,
            capsize=0,
            alpha=0.7,
            label='pp data' if i == 0 else ""
        )

    # ---------------- ensemble ----------------
    mean_pred, std_pred, _ = ensemble_dsdt(replica_models, E, t_dense)

    y_mean = mean_pred * scale_factor
    y_std  = std_pred * scale_factor

    y_low  = np.maximum(y_mean - y_std, 1e-30)
    y_high = np.maximum(y_mean + y_std, 1e-30)

    ax.plot(
        t_dense,
        y_mean,
        'k-',
        linewidth=2,
        label='Ensemble mean' if i == 0 else ""
    )

    ax.fill_between(
        t_dense,
        y_low,
        y_high,
        color='gray',
        alpha=0.25,
        label=r'Ensemble $1\sigma$' if i == 0 else ""
    )

    # ---------------- texto da energia ----------------
    text_energy = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'
    idx_text = np.argmin(np.abs(t_dense - 3.5))
    y_text = max(y_mean[idx_text] * 1.5, 1e-30)

    ax.text(
        3.5,
        y_text,
        text_energy,
        fontsize=10,
        fontweight='bold',
        verticalalignment='bottom'
    )

# ---------------- formatação ----------------
ax.set_yscale('log')
ax.set_xlim(0, 6)

ax.set_xlabel(r'$-t\;(\mathrm{GeV}^2)$', fontsize=14)
ax.set_ylabel(r'$d\sigma/dt\;(\mathrm{mb/GeV}^2)$', fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
ax.tick_params(axis='both', which='minor', direction='in', length=3)

ax.grid(True, which='both', ls='-', alpha=0.1)
ax.legend(loc='upper right', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("ensemble_dsigma_dt.png", dpi=300)
plt.show()

# -------------------------------------------------------------
# PLOT DOS RESÍDUOS DA REDE NEURAL (MÉDIA + BANDA DO ENSEMBLE)
# -------------------------------------------------------------

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica encontrada: modelo_replica_*.pth")

plot_energies = [62.5, 7000.0, 13000.0]
t_eval = np.linspace(0.001, 10.0, 500)
Delta_eval_np = np.sqrt(t_eval).astype(np.float32)

# tensor físico de Delta
Delta_tc_phys = tc.tensor(Delta_eval_np, dtype=tc.float32, device=device).view(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for E in plot_energies:
    s_val = E**2
    s_tc_phys = tc.full_like(Delta_tc_phys, s_val)

    all_R = []
    all_I = []

    with tc.inference_mode():
        for m in replica_models:
            # normalização interna correta
            log_s_z, Delta_z = m._normalize(s_tc_phys, Delta_tc_phys)

            # saída da rede residual
            nn_out = m.nn_amp(log_s_z, Delta_z)

            res_R = nn_out[:, 0].cpu().numpy()
            res_I = nn_out[:, 1].cpu().numpy()

            all_R.append(res_R)
            all_I.append(res_I)

    all_R = np.array(all_R)  # [n_rep, n_t]
    all_I = np.array(all_I)  # [n_rep, n_t]

    # modulação física consistente com sua amplitude
    modulation = Delta_eval_np / E

    all_R_mod = all_R * modulation
    all_I_mod = all_I * modulation

    mean_R = all_R_mod.mean(axis=0)
    mean_I = all_I_mod.mean(axis=0)

    std_R = all_R_mod.std(axis=0, ddof=1) if all_R_mod.shape[0] > 1 else np.zeros_like(mean_R)
    std_I = all_I_mod.std(axis=0, ddof=1) if all_I_mod.shape[0] > 1 else np.zeros_like(mean_I)

    label = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'

    # parte real
    ax1.plot(t_eval, mean_R, linewidth=2, label=label)
    ax1.fill_between(t_eval, mean_R - std_R, mean_R + std_R, alpha=0.25)

    # parte imaginária
    ax2.plot(t_eval, mean_I, linewidth=2, label=label)
    ax2.fill_between(t_eval, mean_I - std_I, mean_I + std_I, alpha=0.25)

# --- formatação eixo real ---
ax1.axhline(0.0, color='black', lw=1, ls='--')
ax1.set_title(r'Real Residual ($\Delta \cdot \mathrm{NN}_R / \sqrt{s}$)', fontsize=14)
ax1.set_xlabel(r'$|t| \, (\mathrm{GeV}^2)$', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, alpha=0.2)
ax1.legend()

# --- formatação eixo imaginário ---
ax2.axhline(0.0, color='black', lw=1, ls='--')
ax2.set_title(r'Imaginary Residual ($\Delta \cdot \mathrm{NN}_I / \sqrt{s}$)', fontsize=14)
ax2.set_xlabel(r'$|t| \, (\mathrm{GeV}^2)$', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True, alpha=0.2)
ax2.legend()

plt.tight_layout()
plt.savefig("nn_residuals_ensemble.png", dpi=300)
plt.show()

# -------------------------------------------------------------
# INTEGRATED CROSS SECTIONS VS ENERGY (ENSEMBLE EXTRAPOLATION)
# -------------------------------------------------------------

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica encontrada: modelo_replica_*.pth")

# Experimental literature data (Inelastic cross-section)
x_lit = np.array([204.25, 906.2, 1968.4, 7350.0, 12481.0, 5.746e4, 1.0505e5], dtype=float)
y_lit = np.array([41.489, 50.337, 57.5, 70.983, 73.23, 91.208, 105.253], dtype=float)

# Energy range in GeV
sqrt_s_range = np.logspace(1, 6, 100)

# Storage for all replicas
stot_all = []
sel_all = []
sinel_all = []

print(f"Calculating integrated cross-sections for {len(replica_models)} replicas...")

with tc.inference_mode():
    for m in replica_models:
        m.eval()

        stot_rep = []
        sel_rep = []
        sinel_rep = []

        for sqs in sqrt_s_range:
            s_tc = tc.tensor([[sqs**2]], dtype=tc.float32, device=device)

            sig_el, sig_inel, sig_tot, _ = m.sigmas(s_tc, mode="pp")

            sel_rep.append(sig_el.item())
            sinel_rep.append(sig_inel.item())
            stot_rep.append(sig_tot.item())

        sel_all.append(sel_rep)
        sinel_all.append(sinel_rep)
        stot_all.append(stot_rep)

# Convert to numpy arrays
sel_all = np.asarray(sel_all, dtype=float)
sinel_all = np.asarray(sinel_all, dtype=float)
stot_all = np.asarray(stot_all, dtype=float)

# Ensemble mean
sel_mean = sel_all.mean(axis=0)
sinel_mean = sinel_all.mean(axis=0)
stot_mean = stot_all.mean(axis=0)

# Ensemble std
if len(replica_models) > 1:
    sel_std = sel_all.std(axis=0, ddof=1)
    sinel_std = sinel_all.std(axis=0, ddof=1)
    stot_std = stot_all.std(axis=0, ddof=1)
else:
    sel_std = np.zeros_like(sel_mean)
    sinel_std = np.zeros_like(sinel_mean)
    stot_std = np.zeros_like(stot_mean)

# -------------------------------------------------------------
# FINAL EXTRAPOLATION PLOT
# -------------------------------------------------------------

plt.figure(figsize=(10, 7))

def plot_ensemble_sigma(x, mean, std, color, label, ls='-'):
    plt.plot(x, mean, color=color, ls=ls, lw=2, label=label)
    plt.fill_between(
        x,
        np.maximum(mean - std, 0.0),
        mean + std,
        color=color,
        alpha=0.15
    )

plot_ensemble_sigma(sqrt_s_range, stot_mean, stot_std, 'black', r'$\sigma_{\mathrm{tot}}$ (ensemble)', ls='-')
plot_ensemble_sigma(sqrt_s_range, sinel_mean, sinel_std, 'red', r'$\sigma_{\mathrm{inel}}$ (ensemble)', ls='--')
plot_ensemble_sigma(sqrt_s_range, sel_mean, sel_std, 'blue', r'$\sigma_{\mathrm{el}}$ (ensemble)', ls=':')

# Literature points
plt.scatter(
    x_lit, y_lit,
    color='darkred',
    marker='s',
    s=40,
    label=r'$\sigma_{\mathrm{inel}}$ (literature)',
    zorder=5
)

# Formatting
plt.xscale('log')
plt.xlim(10, 1e6)
plt.ylim(10, 350)

plt.xlabel(r'$\sqrt{s}\,(\mathrm{GeV})$', fontsize=14)
plt.ylabel(r'$\sigma\,(\mathrm{mb})$', fontsize=14)
plt.title('Proton-Proton Cross Sections Extrapolation', fontsize=16)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12, frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig("integrated_cross_sections_ensemble.png", dpi=300)
plt.show()