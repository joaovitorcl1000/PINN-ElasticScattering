import numpy as np
import pandas as pd
import os
import random
import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from dataclasses import dataclass

#number of threads used
tc.set_num_threads(14)

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
    log_Delta_mean: tc.Tensor
    log_Delta_std : tc.Tensor
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
        cols_req = ["sqrt_s_GeV", "t_GeV2", "dsig_dt_mb_GeV2", "mode"]
        df_fit = df.dropna(subset=cols_req).copy()
        # Filter out unphysical or zero cross-sections
        df_fit = df_fit[df_fit["dsig_dt_mb_GeV2"] > 1e-15]
        self.df_fit = df_fit
        self.mode_raw = df_fit["mode"].values # pp or pbarp

        # --- Physics Variable Transformation ---
        # Convert to float32 early to save memory and ensure torch compatibility
        dsig_dt = df_fit["dsig_dt_mb_GeV2"].values.astype(np.float32)
        sqrt_s  = df_fit["sqrt_s_GeV"].values.astype(np.float32)
        t_abs   = np.abs(df_fit["t_GeV2"].values.astype(np.float32))

        # Log-scale is preferred for cross-sections due to high dynamic range
        s_pp = sqrt_s **2
        log_s = np.log(s_pp).astype(np.float32)
        Delta = np.sqrt(t_abs)
        log_Delta = np.log(np.sqrt(t_abs) + 1e-9)
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
        self.log_Delta_tc     = prep(log_Delta)
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
            log_Delta_mean = self.log_Delta_tc.mean(),
            log_Delta_std  = self.log_Delta_tc.std() + 1e-12,
            Y_mean     = self.Y_tc.mean(),
            Y_std      = self.Y_tc.std() + 1e-12
        )
        
        # --- Shortcut attributes (Pointing to the same memory) ---
        # This satisfies your _normalize method without recalculating anything
        self.log_s_tc_mean = self.stats.log_s_mean
        self.log_s_tc_std  = self.stats.log_s_std
        self.Delta_tc_mean = self.stats.Delta_mean
        self.Delta_tc_std  = self.stats.Delta_std
        self.log_Delta_tc_mean = self.stats.log_Delta_mean
        self.log_Delta_tc_std  = self.stats.log_Delta_std

#--------------------------------------------------------------------------------------------------    

#Loads data globally
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

# Instantiate the class by loading the CSV.
data = ScatteringData(csv_path="data.csv", device=device)

# =============================================================================
# 3. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class NNAmplitude(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        # Input Layer
        self.input_layer = nn.Linear(2, hidden_dim)
        
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Smooth activation
        self.activation = nn.SiLU() 
        
        # Output Layer (Real and Imaginary part)
        self.output_layer = nn.Linear(hidden_dim, 2)
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        
        # We begin with zero at output
        nn.init.constant_(self.output_layer.weight, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, log_s_z, log_Delta_z):
        x = tc.cat([log_s_z, log_Delta_z], dim=1)
        
        #Layers
        x = self.activation(self.input_layer(x))
        
        identity = x
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x + identity)) 
        
        identity = x
        x = self.activation(self.layer3(x))
        x = x + identity 
        
        return self.output_layer(x)
    
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
        self.register_buffer("log_Delta_mean", scattering_instance.log_Delta_tc_mean)
        self.register_buffer("log_Delta_std",  scattering_instance.log_Delta_tc_std)

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
        
        self.n1 = nn.Parameter(tc.tensor([1e0]))   # Growth rate with energy (Dimensionless)
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
        Normalizes the entries of the NN
        """
        if s.dim() == 1: s = s.view(-1, 1)
        if Delta.dim() == 1: Delta = Delta.view(-1, 1)
        
        ln_s = tc.log(s)
        log_Delta = tc.log(Delta)
        
        ln_s_z = (ln_s - self.log_s_mean) / self.log_s_std
        log_Delta_z = (log_Delta - self.log_Delta_mean) / self.log_Delta_std
        
        return ln_s_z, log_Delta_z
    
    # ---------------- sigma_tot(s) ----------------------------------
    # COMPETE PARAMETERIZATION (Phys. Rev. Lett 89 (2002) 201801)
    # ----------------------------------------------------------------

    def sigma_tot(self, s, mode):
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

    def rho(self, s, mode):
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
        factor = -1.0 if mode == "pp" else 1.0
            
        # pp: (even) - (odd)  |  pbarp: (even) + (odd)    
        sig = (pomeron + regge_p) + (factor * regge_m)
        d_sig = (d_pomeron + d_regge_p) + (factor * d_regge_m)
            
        # DDR Formula
        # We use a small epsilon for numerical safety
        rho_val = (tc.pi / 2.0) * (d_sig / (sig + 1e-9))
            
        return tc.clamp(rho_val, min=-0.5, max=0.5)
    
    # ----------------------------------------------------------------
    # ---------------- Elastic Amplitude -----------------------------
    # ----------------- A_el(s, Delta) -------------------------------
    # ----------------------------------------------------------------

    def A_el(self, s, Delta, mode):
        """
        Calculates the Real and Imaginary components of the amplitude.
        This is the "brain" of the model.
        """
        # A. Physical Scales Conversion
        ln_s_z, log_Delta_z = self._normalize(s, Delta)
        ln_s = tc.log(s)
        
        # B. Neural Network Residuals
        nn_out = self.nn_amp(ln_s_z, log_Delta_z)
        out_R, out_I = nn_out[:, 0:1], nn_out[:, 1:2]

        # C. Physics Baseline (COMPETE + Dipole)
        sig_tot = self.sigma_tot(s, mode)
        rho_val = self.rho(s, mode)

        # D. Assembling the Amplitude with NN Corrections
        energy_suppr = tc.sqrt(s)

        support = (Delta / energy_suppr)

        f_R = out_R * support 
        f_I = out_I * support

        # Real (F_R) and Imaginary (F_I) profile components
        n_eff = tc.clamp(self.n0 + tc.abs(self.n1) * tc.exp(-tc.abs(self.epsilon) * (ln_s - self.log_s0)), min=1.0)
        B_s   = tc.clamp(self.B0 + 2.0 * self.alpha_prime * (ln_s - self.log_s_regge), min=1e-3)

        base = tc.clamp(1.0 + (B_s * Delta**2) / n_eff, min=1e-7)
        F_dip = tc.exp(-n_eff * tc.log(base))

        F_R = (rho_val + f_R) * F_dip
        F_I = (1.0 + f_I) * F_dip
        
        Re_A_el = s * sig_tot * F_R  
        Im_A_el = s * sig_tot * F_I

        return Re_A_el, Im_A_el

    # ----------------------------------------------------------------
    # Differential Elastic Cross Section
    # ----------------------------------------------------------------

    def dsigma_dt(self, s, Delta, mode):
        """
        Orchestrates the amplitude calculation and returns dsigma/dt.
        """
        # Get the components from the amplitude method
        Re_A_el, Im_A_el = self.A_el(s, Delta, mode)

        # Final Calculation: dsigma/dt = (sig_tot^2 / 16*pi*hbarc2) * |F|^2
        mod_A2 = Re_A_el**2 + Im_A_el**2
        dsigma_dt = mod_A2 / (16.0 * tc.pi * self.hbarc2 * s **2)
        
        return dsigma_dt

    # ----------------------------------------------------------------
    # Cross Sections
    # ----------------------------------------------------------------

    def sigmas(self, s, mode):
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
        dsig_dt_eval = self.dsigma_dt(s_eval, Delta_eval, mode)
        
        # 6. Reshape back to [Energies, t_points]
        dsig_dt_2d = dsig_dt_eval.view(num_energies, -1)

        # 7. Integration (FIX: Using the buffer t_grid here)
        sig_el = tc.trapezoid(dsig_dt_2d, t_grid, dim=1).view(-1, 1)

        # 8. Analytical sigma_tot
        sigma_tot = self.sigma_tot(s_flat, mode).view(-1, 1)
        
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

# =============================================================================
# UNCERTAINTY BAND
# =============================================================================

def ensemble_dsdt(replica_models, sqrt_s, t_grid):
    """
    It calculates the mean and uncertainty of the ensemble on a log scale.
    """
    all_preds = []
    device = next(replica_models[0].parameters()).device
    
    s_val = tc.tensor([[sqrt_s**2]], dtype=tc.float32, device=device)
    t_tc = tc.tensor(t_grid, dtype=tc.float32, device=device).view(-1, 1)
    s_tc = s_val.expand(t_tc.size(0), 1)
    Delta_tc = tc.sqrt(t_tc)

    with tc.inference_mode():
        for m in replica_models:
            m.eval()
            pred = m.dsigma_dt(s_tc, Delta_tc, mode="pp").cpu().numpy().flatten()
            all_preds.append(np.log(pred + 1e-30))

    all_preds = np.array(all_preds) # Shape: [n_replicas, n_t]
    
    log_mean = all_preds.mean(axis=0)
    log_std = all_preds.std(axis=0, ddof=1)
    
    mean_val = np.exp(log_mean)
    low_bound = np.exp(log_mean - log_std)
    high_bound = np.exp(log_mean + log_std)
    
    return mean_val, low_bound, high_bound, all_preds

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


# ensemble mean model
best_model = EnsembleMeanModel(replica_models).eval()

# -------------------------------------------------------------
# PHYSICAL PARAMETERS (arithmetic mean)
# -------------------------------------------------------------

n0_vals = np.array([m.n0.item() for m in replica_models])
n1_vals = np.array([m.n1.item() for m in replica_models])
ep_vals = np.array([m.epsilon.item() for m in replica_models])

n0_std = n0_vals.std(ddof=1) if len(n0_vals) > 1 else 0.0
n1_std = n1_vals.std(ddof=1) if len(n1_vals) > 1 else 0.0
ep_std = ep_vals.std(ddof=1) if len(ep_vals) > 1 else 0.0

print("\nENSEMBLE PHYICAL PARAMETERS:")
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
# dsigma/dt GRAPH WITH ENSEMBLE MEAN + UNCERTAINTY BAND
# -------------------------------------------------------------

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica encontrada: modelo_replica_*.pth")

target_energies = [23.5, 30.7, 44.7, 52.8, 62.5, 7000.0, 13000.0]
t_dense = np.linspace(0.001, 10.0, 500)

fig, ax = plt.subplots(figsize=(10, 10))

for i, E in enumerate(target_energies):
    scale_factor = 10**(-2 * i)

    # ---------------- experimental data ----------------
    tol = 0.5 if E < 100 else 50.0
    mask = (data.df_fit['sqrt_s_GeV'] > E - tol) & (data.df_fit['sqrt_s_GeV'] < E + tol)
    df_E = data.df_fit[mask]

    if len(df_E) > 0:
        x_data = np.abs(df_E['t_GeV2'].values)
        y_data = df_E['dsig_dt_mb_GeV2'].values * scale_factor
        err_p = np.abs(df_E['err_total_plus'].values) * scale_factor
        err_m = np.abs(df_E['err_total_minus'].values) * scale_factor
        err_m = np.minimum(err_m, 0.9999 * y_data)

        ax.errorbar(x_data, y_data, yerr=[err_m, err_p], fmt='o', color='red',
                    markersize=3, capsize=0, alpha=0.7, label='pp data' if i == 0 else "")

    # ---------------- ensemble (LOG-STAT) ----------------
    all_log_preds = []
    with tc.inference_mode():
        s_val = tc.tensor([[E**2]], dtype=tc.float32, device=device)
        t_tc = tc.tensor(t_dense, dtype=tc.float32, device=device).view(-1, 1)
        Delta_tc = tc.sqrt(t_tc)
        for m in replica_models:
            pred = m.dsigma_dt(s_val.expand(t_tc.size(0), 1), Delta_tc, mode="pp")
            all_log_preds.append(tc.log(pred + 1e-35).cpu().numpy().flatten())
    
    all_log_preds = np.array(all_log_preds)
    log_mean = all_log_preds.mean(axis=0)
    log_std  = all_log_preds.std(axis=0, ddof=1)

    y_mean = np.exp(log_mean) * scale_factor
    y_low  = np.exp(log_mean - log_std) * scale_factor
    y_high = np.exp(log_mean + log_std) * scale_factor

    ax.plot(t_dense, y_mean, 'k-', linewidth=2, label='Ensemble mean' if i == 0 else "")

    ax.fill_between(t_dense, y_low, y_high, color='gray', alpha=0.25,
                    label=r'Ensemble $1\sigma$ (Log-Stat)' if i == 0 else "")

    text_energy = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'
    idx_text = np.argmin(np.abs(t_dense - 3.5))
    y_text = max(y_mean[idx_text] * 1.5, 1e-30)

    ax.text(3.5, y_text, text_energy, fontsize=10, fontweight='bold', verticalalignment='bottom')

ax.set_yscale('log')
ax.set_xlim(0, 10)
ax.set_xlabel(r'$-t\;(\mathrm{GeV}^2)$', fontsize=14)
ax.set_ylabel(r'$d\sigma/dt\;(\mathrm{mb/GeV}^2)$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
ax.tick_params(axis='both', which='minor', direction='in', length=3)
ax.grid(True, which='both', ls='-', alpha=0.1)
ax.legend(loc='upper right', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("outputs/plots/ensemble_dsigma_dt.png", dpi=300)
plt.show()

# =============================================================================
# PLOT OF NEURAL NETWORK RESIDUES (AVERAGE + ENSEMBLE BAND)
# =============================================================================

if len(replica_models) == 0:
    raise RuntimeError("Nenhuma réplica encontrada. Carregue os modelos .pth primeiro.")

plot_energies = [62.5, 7000.0, 13000.0]
t_eval = np.linspace(0.001, 10.0, 500)
Delta_eval_np = np.sqrt(t_eval).astype(np.float32)

Delta_tc_phys = tc.tensor(Delta_eval_np, dtype=tc.float32, device=device).view(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for E in plot_energies:
    s_val = E**2
    s_tc_phys = tc.full_like(Delta_tc_phys, s_val)

    all_R = []
    all_I = []

    with tc.inference_mode():
        for m in replica_models:
            m.eval() 
            
            log_s_z, Delta_z = m._normalize(s_tc_phys, Delta_tc_phys)

            nn_out = m.nn_amp(log_s_z, Delta_z)

            all_R.append(nn_out[:, 0].cpu().numpy())
            all_I.append(nn_out[:, 1].cpu().numpy())

    all_R = np.array(all_R)
    all_I = np.array(all_I)

    # Physical Modelling: f = (Delta / sqrt(s)) * NN_output
    # This makes the residue comparable across different energy levels.
    modulation = Delta_eval_np / E
    all_R_mod = all_R * modulation
    all_I_mod = all_I * modulation

    # Ensemble Statistic (Mean and Standard Deviation)
    mean_R = all_R_mod.mean(axis=0)
    std_R  = all_R_mod.std(axis=0, ddof=1) if len(replica_models) > 1 else 0
    
    mean_I = all_I_mod.mean(axis=0)
    std_I  = all_I_mod.std(axis=0, ddof=1) if len(replica_models) > 1 else 0

    label = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'

    # --- Plot Real component ---
    ax1.plot(t_eval, mean_R, linewidth=2, label=label)
    ax1.fill_between(t_eval, mean_R - std_R, mean_R + std_R, alpha=0.2)

    # --- Plot Imaginary component ---
    ax2.plot(t_eval, mean_I, linewidth=2, label=label)
    ax2.fill_between(t_eval, mean_I - std_I, mean_I + std_I, alpha=0.2)

ax1.axhline(0.0, color='black', lw=1, ls='--')
ax2.axhline(0.0, color='black', lw=1, ls='--')

ax1.set_title(r'Real Residual ($f_R$)', fontsize=14)
ax2.set_title(r'Imaginary Residual ($f_I$)', fontsize=14)

for ax in [ax1, ax2]:
    ax.set_xlabel(r'$|t| \, (\mathrm{GeV}^2)$', fontsize=12)
    ax.set_ylabel('Amplitude Correction', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend()

plt.tight_layout()
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
plt.savefig("outputs/plots/integrated_cross_sections_ensemble.png", dpi=300)
plt.show()
