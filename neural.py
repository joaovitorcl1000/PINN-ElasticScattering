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

tc.set_num_threads(14)

#--------------------------------------------------------------------------------------------------

num_replicas = 1    # Number of MC Replicas
n_epochs = 1000000     # Number of epochs
target_loss = 5e1     # Stop training, it's a great loss.

patience_limit = 10000  # Wait that long; if the learning stagnates, return to the best value of the loss (jump).
threshold = 1.0e2       # It only saves if the loss (cost function) is less than that.
max_jumps = 3           # Limit of times the network can "jump" before Early Stopping.

replica_fraction = 0.8 # Fraction of data used in the train

MC_Central_Y_fluctuation = True # Gaussian 1sigma fluctuation around central value in the replicas

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

seed_choice = fix_seeds(42)

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

        subset.mode_raw = self.mode_raw[indices.cpu().numpy()] if tc.is_tensor(indices) else self.mode_raw[indices]
        
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

    def generate_replica_split(self, split_ratio=replica_fraction):
        """
        Generates a Monte Carlo replica by fluctuating data points within 
        experimental errors and splitting by energy levels (sqrt_s).
        """
        
        if MC_Central_Y_fluctuation:
            # 1. Generate Gaussian fluctuations (Asymmetric)
            z_random = tc.randn_like(self.Y_tc)
            # Apply plus or mnus error based on the sign of the random shift
            fluctuation = tc.where(z_random > 0, 
                                    z_random * self.Err_log_p_tc, 
                                    z_random * self.Err_log_m_tc)
        else:
            fluctuation = 0.0 #central value
        
        # Create a temporary clone for the full replica values
        y_replica_values = self.Y_tc.clone() + fluctuation

        # 2. Energy-based splitting (Golden Rule for Scattering Data)
        sqrt_s_flat = self.sqrt_s_tc.view(-1)
        unique_energy = tc.unique(sqrt_s_flat)
        
        n_unique = unique_energy.numel()
        shuffled_idx = tc.randperm(n_unique, device=self.device) # Alternative without shuffle the data tc.arange(n_unique, device=self.device)

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
        self.n0.requires_grad_(True)
        
        self.n1 = nn.Parameter(tc.tensor([1e-3]))   # Growth rate with energy (Dimensionless)
        self.n1.requires_grad_(True)           

        self.epsilon = nn.Parameter(tc.tensor([0.05])) 
        self.epsilon.requires_grad_(False)             

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
        ln_s_z, Delta_z = self._normalize(s, Delta)
        ln_s = tc.log(s)
        
        # B. Neural Network Residuals
        nn_out = self.nn_amp(ln_s_z, Delta_z)
        out_R, out_I = nn_out[:, 0:1], nn_out[:, 1:2]

        # C. Physics Baseline (COMPETE + Dipole)
        sig_tot = self.sigma_tot(s, mode)
        rho_val = self.rho(s, mode)
        
        n_eff = self.n0 + self.n1 * (ln_s - self.log_s0)
         #tc.clamp(self.n0 + tc.abs(self.n1) * tc.exp(-tc.abs(self.epsilon) * (ln_s - self.log_s0)), min=1e-3)
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
# Cost Function (Chi2 + Physics Informed Penalties)
# =============================================================================

def chi2(y_pred, data):
    """
    Computes the statistical Chi2 given the predicted values and the dataset.
    This avoids re-running the Neural Network multiple times.
    """
    # Residual: Prediction - Target
    residuals = y_pred - data.Y_tc
    
    # Asymmetric errors: 
    # If residual >= 0, prediction is above data -> use Err_plus
    # If residual < 0, prediction is below data -> use Err_mnus
    Err_eff = tc.where(residuals >= 0, data.Err_log_p_tc, data.Err_log_m_tc)
    
    chi2_val = tc.mean((residuals / (Err_eff + 1e-12))**2)
    
    return chi2_val

def cost_function(model, data, epoch):
    """
    Calculates the total loss: Statistical Chi2 + Curvature Physics + Unitarity.
    """
    # 1. Prepare Data
    # Calculate s from sqrt_s. We need s to be a tensor for the model.
    s_tc = data.s_tc
    
    # We must explicitly tell PyTorch to track gradients for Delta 
    # to compute the curvature penalty later.
    Delta_tc = data.Delta_tc.clone().requires_grad_(True)

    mask_pp = (data.mode_raw == "pp")
    mask_pbarp = (data.mode_raw == "pbarp")

    # We initialize with the same size of data
    # Prediction: model directly returns dsigma_dt in physical units
    y_pred = tc.zeros_like(data.Y_tc)
    
    # pp
    if np.any(mask_pp):
        pred_pp = model.dsigma_dt(s_tc[mask_pp], Delta_tc[mask_pp], "pp")
        # We convert the prediction to log space, as Y_tc = ln(dsig_dt_data)
        y_pred[mask_pp] = tc.log(tc.clamp(pred_pp, min=1e-30))

    # pbarp
    if np.any(mask_pbarp):
        pred_pb = model.dsigma_dt(s_tc[mask_pbarp], Delta_tc[mask_pbarp], "pbarp")
        # We convert the prediction to log space, as Y_tc = ln(dsig_dt_data)
        y_pred[mask_pbarp] = tc.log(tc.clamp(pred_pb, min=1e-30))

    chi2_val = chi2(y_pred, data)

    # -------------------------------------------------------------------------
    # 3. Curvature Penalty (Second derivative wrt Delta)
    # -------------------------------------------------------------------------
    first_derivative = tc.autograd.grad(
        outputs=y_pred,
        inputs=Delta_tc,             # Taking derivative with respect to Delta
        grad_outputs=tc.ones_like(y_pred),
        create_graph=True,
        retain_graph=True
    )[0]

    second_derivative = tc.autograd.grad(
        outputs=first_derivative,
        inputs=Delta_tc,             # Taking derivative of the derivative
        grad_outputs=tc.ones_like(first_derivative),
        create_graph=True,
        retain_graph=True
    )[0]

    lambda_curv = 1e-3
    loss_curv = lambda_curv * tc.mean(second_derivative**2)

    # -------------------------------------------------------------------------
    # 4. Unitarity Penalty (sigma_el <= sigma_tot)
    # -------------------------------------------------------------------------

    loss_unitarity = 0.0

    if epoch % 20 == 0:
        if np.any(mask_pp):
            s_unique_pp = tc.unique(s_tc[mask_pp])
            _, _, _, ratio_pp = model.sigmas(s_unique_pp, "pp")
            loss_unitarity = loss_unitarity + tc.sum(tc.relu(ratio_pp - 1.0))

        if np.any(mask_pbarp):
            s_unique_pbarp = tc.unique(s_tc[mask_pbarp])
            _, _, _, ratio_pbarp = model.sigmas(s_unique_pbarp, "pbarp")
            loss_unitarity = loss_unitarity + tc.sum(tc.relu(ratio_pbarp - 1.0))

    # If ratio > 1.0, (ratio - 1.0) is positive and ReLU keeps it (Penalizing).
    # If ratio <= 1.0, (ratio - 1.0) is negative and ReLU zeroes it (All good!).
    lu = 1e-1 * loss_unitarity

    # -------------------------------------------------------------------------
    # Final Loss
    # -------------------------------------------------------------------------

    total_loss = chi2_val + loss_curv + lu
    
    return total_loss, chi2_val

#--------------------------------------------------------------------------------------------------    

# =============================================================================
# Network Training - Monte Carlo Training Loop (100 Replicas)
# =============================================================================

trained_models = []
replica_statistics = []
approved_val_losses = []

# Helper function to properly reset optimizer and scheduler during "jumps"
def get_optim_sched(model, lr=5e-4):
    opt = tc.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = tc.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2000, T_mult=1)
    return opt, sched

for rep in range(num_replicas):
    print(f"\n" + "="*60)
    print(f"🚀 STARTING REPLICA {rep+1}/{num_replicas}")
    print("="*60)

    # 1. Generate fluctuated and split data for this replica
    data_train, data_val = data.generate_replica_split(split_ratio=0.8)

    # 2. Instantiate the network FROM SCRATCH to ensure independence
    model = PhysicalAmplitude(data).to(device)

    # if os.path.exists('best_model.pth'):
    #     checkpoint = tc.load('best_model.pth', map_location=device)
    #     # Remove o prefixo '_orig_mod.' se ele existir no arquivo salvo
    #     clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    #     model.load_state_dict(clean_state_dict)
    #     print("✅ Pre-trained weights loaded and cleaned.")

    # Initialize Optimizer and Scheduler
    optimizer, scheduler = get_optim_sched(model, lr=5e-4)

    loss_hist = []
    best_val_loss = float('inf')
    trigger_times = 0
    jumps_made = 0
    
    # Unique temp name to avoid thread collisions
    temp_name = f'modelo_temp_rep_{rep+1:03d}.pth'
    final_name = f'modelo_replica_{rep+1:03d}.pth'

    for ep in range(n_epochs):
        # --- TRAINING PHASE (Only on 80% subset) ---
        model.train()
        optimizer.zero_grad(set_to_none=True)        

        # Unpack the two returns from our updated cost_function
        loss_train, chi2_train = cost_function(model, data_train, ep)

        if tc.isnan(loss_train):
            print(f"💥 NAN DETECTED at epoch {ep}. Aborting this replica.")
            break # Saída imediata do loop de épocas desta réplica
        
        # Backpropagation MUST be done only on the total_loss
        loss_train.backward()

        tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # --- VALIDATION PHASE (Only on 20% subset) ---
        model.eval()
        # We can safely use no_grad() because we calculate pure math here!
        with tc.inference_mode():
            # 1. Prepare inputs
            s_val = data_val.s_tc
            Delta_val = data_val.Delta_tc

            mask_pp = (data_val.mode_raw == "pp")
            mask_pbarp = (data_val.mode_raw == "pbarp")

            y_pred_val = tc.zeros_like(data_val.Y_tc)
            # 2. Single forward pass without calculating physics gradients

            if mask_pp.any():
                pred_pp = model.dsigma_dt(s_val[mask_pp], Delta_val[mask_pp], "pp")
                y_pred_val[mask_pp] = tc.log(tc.clamp(pred_pp, min=1e-30))

            if mask_pbarp.any():
                pred_pbarp = model.dsigma_dt(s_val[mask_pbarp], Delta_val[mask_pbarp], "pbarp")
                y_pred_val[mask_pbarp] = tc.log(tc.clamp(pred_pbarp, min=1e-30))

            loss_val = chi2(y_pred_val, data_val).item()
            
            # 3. Compute statistical Chi2
            loss_val = chi2(y_pred_val, data_val).item()

        loss_hist.append(loss_val)

        # Update the best model if validation improves
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            # --- MUDANÇA AQUI: Salva sempre o modelo original ---
            state_to_save = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            tc.save(state_to_save, temp_name)
            trigger_times = 0

        else:
            trigger_times += 1

        # Stagnation and "Jump" Logic
        if trigger_times >= patience_limit:
            if jumps_made < max_jumps:
                print(f"🔄 Validation stagnated. Jumping (Jump {jumps_made+1}/{max_jumps})...")
                if os.path.exists(temp_name):
                    model.load_state_dict(tc.load(temp_name))
                
                # We completely rebuild the optimizer and scheduler to force the new LR
                optimizer, scheduler = get_optim_sched(model, lr=5e-3)
                
                trigger_times = 0
                jumps_made += 1
            else:
                print(f"🛑 Jump limit reached. Finishing replica by Early Stopping.")
                break

        # Occasional Prints tracking both Total Loss and Pure Chi2
        if ep % 500 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"[{ep:5d}] Loss Tot: {loss_train.item():.2e} | Chi2 Train: {chi2_train.item():.2e} | Chi2 Val: {loss_val:.2e} | Best: {best_val_loss:.2e} | LR: {curr_lr:.1e}")

        # Premature Victory Condition
        if (loss_val <= target_loss) and (loss_train.item() <= target_loss):
            print(f"🎯 Target loss achieved on validation!")
            tc.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), temp_name)
            break

    # -------------------------------------------------------------
    # --- REPLICA EVALUATION AND SAVING CRITERIA ---
    # -------------------------------------------------------------
    
    # Recover the best weights saved in the temporary file
    if os.path.exists(temp_name):
        # Load back to the model before any evaluation or final save
        checkpoint = tc.load(temp_name, map_location=device)
        # Handle compiled model prefix if necessary
        model.load_state_dict(checkpoint)
        os.remove(temp_name) 

    # 2. Check if this BEST version is approved
    if best_val_loss <= threshold:
        # --- MUDANÇA AQUI: Adiciona à lista de aprovados para evitar o erro do argmin ---
        approved_val_losses.append(best_val_loss)
        
        # Salva o arquivo final da réplica (limpo)
        state_to_save = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        tc.save(state_to_save, final_name)
        
        # Atualiza o best_model.pth global se for o melhor de todos
        # Note: 'global_best_loss' deve ser definida como float('inf') antes do loop das réplicas
        if 'global_best_loss' not in locals() or best_val_loss < global_best_loss:
            global_best_loss = best_val_loss
            tc.save(state_to_save, 'best_model.pth')
            
        trained_models.append(copy.deepcopy(model))
        
        # --- PHYSICAL EVALUATION AGAINST REAL DATA ---
        model.eval()
        with tc.no_grad():
            # Evaluate using the original `data` object (without error fluctuation)
            s_full = data.s_tc
            Delta_full = data.Delta_tc

            mask_pp = (data.mode_raw == "pp")
            mask_pbarp = (data.mode_raw == "pbarp")

            y_pred_full = tc.zeros_like(data.Y_tc)

            if mask_pp.any():
                pred_pp = model.dsigma_dt(s_full[mask_pp], Delta_full[mask_pp], "pp")
                y_pred_full[mask_pp] = tc.log(tc.clamp(pred_pp, min=1e-30))

            if mask_pbarp.any():
                pred_pbarp = model.dsigma_dt(s_full[mask_pbarp], Delta_full[mask_pbarp], "pbarp")
                y_pred_full[mask_pbarp] = tc.log(tc.clamp(pred_pbarp, min=1e-30))

            mean_loss = chi2(y_pred_full, data)

            n_data = data.Y_tc.shape[0]
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            dof = n_data - n_params # Degrees of freedom

            chi2_total = mean_loss.item() * n_data
            chi2_red = chi2_total / dof if dof > 0 else float('inf')

            print(f"📊 Total Chi2: {chi2_total:.4f} | DOF: {dof} | Reduced Chi2: {chi2_red:.4f}")
            
            # Save statistics for later analysis
            replica_statistics.append(chi2_red)
    else:
        print(f"❌ Replica {rep+1} rejected (Val Loss: {best_val_loss:.4e} > {threshold}). Discarded.")


if len(approved_val_losses) > 0:
    idx_best = int(np.argmin(approved_val_losses))
    best_model = trained_models[idx_best]
    print(f"\n✅ Best model found! Index: {idx_best} | Loss: {approved_val_losses[idx_best]:.4f}")
else:
    print("\n⚠️ No replicas met the threshold. Using the last trained model for plots.")
    best_model = model

print("\n" + "="*60)
print(f"🎉 MONTE CARLO TRAINING FINISHED!")
print(f"Approved models: {len(trained_models)}/{num_replicas}")
print("="*60)

# -------------------------------------------------------------
# DISPLAYING PHYSICAL PARAMETERS OF THE BEST NETWORK
# -------------------------------------------------------------
if len(trained_models) > 0:
    idx_best = int(np.argmin(approved_val_losses))
    best_model = trained_models[idx_best]
else:
    best_model = model

print("\nFOUND PHYSICAL PARAMETERS:")
print(f"n0 fixed:      {best_model.n0.item():.4f}")
print(f"n1 adjusted:   {best_model.n1.item():.4f}")
print(f"ep adjusted:   {best_model.epsilon.item():.4f}")

#--------------------------------------------------------------------------------------------------    

# =============================================================================
# Plotting the Elastic Differential Cross Section (Model vs Data)
# =============================================================================

# 1. Choose the target energies you want to plot (in GeV)
target_energies = [23.5, 30.7, 44.7, 52.8, 62.5, 7000.0, 13000.0]

# 2. Create a dense grid for -t (momentum transfer) to plot smooth theoretical curves
t_dense_np = np.linspace(0.001, 10.0, 500)
Delta_dense_np = np.sqrt(t_dense_np).astype(np.float32)

# Convert the dense grid to a PyTorch tensor on the correct device
Delta_tc_pred = tc.tensor(Delta_dense_np, dtype=tc.float32, device=device).view(-1, 1)

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

for i, E in enumerate(target_energies):
    # --- A. Plotting the Experimental Data ---
    # Tolerance for energy matching (larger tolerance for high energies)
    tol = 0.5 if E < 100 else 50.0
    
    # Filter the original DataFrame for the specific energy
    mask = (
    (data.df_fit['sqrt_s_GeV'] > E - tol) &
    (data.df_fit['sqrt_s_GeV'] < E + tol) &
    (data.df_fit['mode'] == "pp")
    )
    df_E = data.df_fit[mask]

    if len(df_E) > 0:
        x_data = np.abs(df_E['t_GeV2'].values)
        y_data = df_E['dsig_dt_mb_GeV2'].values
        
        # Absolute errors
        err_plus = np.abs(df_E['err_total_plus'].values)
        err_minus = np.abs(df_E['err_total_minus'].values)
        
        # Prevent negative lower error bars on a log scale plot
        err_minus = np.minimum(err_minus, y_data * 0.9999)

        ax.errorbar(
            x_data, y_data,
            yerr=[err_minus, err_plus],
            fmt='o', color='red',
            markersize=4, capsize=0,
            label='pp Data' if i == 0 else ""
        )

    # --- B. Plotting the Theoretical Model ---
    s_val = E**2
    
    # Create a tensor for s with the same shape as the Delta grid
    s_tc_pred = tc.full((len(t_dense_np), 1), s_val, dtype=tc.float32, device=device)

    # Put the best model in evaluation mode
    best_model.eval()
    with tc.no_grad():
        # The model gracefully handles all the physical equations and normalizations!
        dsigma_dt_pred = best_model.dsigma_dt(s_tc_pred, Delta_tc_pred, "pp")
        
        # Bring the prediction back to the CPU as a numpy array for plotting
        y_pred_np = dsigma_dt_pred.cpu().numpy().flatten()

    ax.plot(t_dense_np, y_pred_np, 'k-', linewidth=2, label='NN Model' if i == 0 else "")

    # --- C. Add Energy Labels to the curves ---
    text_energy = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'
    
    # Find a good spot to place the text (e.g., around t = 3.5)
    idx_text = np.argmin(np.abs(t_dense_np - 3.5))
    ax.text(
        3.5,
        y_pred_np[idx_text] * 1.5,
        text_energy,
        fontsize=11,
        fontweight='bold',
        verticalalignment='bottom'
    )

# -------------------------------------------------------------
# 4. Figure Formatting (Scientific Style)
# -------------------------------------------------------------
ax.set_yscale('log')
ax.set_xlim(0, 6)

# Scientific labels using LaTeX formatting
ax.set_xlabel(r'$-t \ (\text{GeV}^2)$', fontsize=14)
ax.set_ylabel(r'$\frac{d\sigma}{dt} \ (\text{mb/GeV}^2)$', fontsize=14)

# Inward pointing ticks, standard for physics publications
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
ax.tick_params(axis='both', which='minor', direction='in', length=3)

# Clean legend
ax.legend(loc='upper right', frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig("Diff_elastic_cross_sections.png", dpi=300) 
plt.show()

# =============================================================================
# Plotting the Neural Network Residuals (Real and Imaginary)
# =============================================================================

best_model.eval()

# Energies of interest
plot_energies = [62.5, 7000.0, 13000.0]
t_eval = np.linspace(0.001, 10.0, 500)
Delta_eval_np = np.sqrt(t_eval).astype(np.float32)

# Transform Delta to a Tensor in physical units
# Using best_model.device ensures everything stays on the same hardware (CPU/GPU)
Delta_tc = tc.tensor(Delta_eval_np, dtype=tc.float32, device=best_model.device).view(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for E in plot_energies:
    # 1. Prepare the physical 's' tensor (Energy squared)
    s_val = E**2
    s_tc = tc.full((len(t_eval), 1), s_val, dtype=tc.float32, device=best_model.device)
    
    # 2. THE MAGIC: Use the model's internal normalizer!
    # Returns Z-scores ready for the Neural Network
    log_s_z, Delta_z = best_model._normalize(s_tc, Delta_tc)

    with tc.no_grad():
        # 3. Direct pass through the internal Neural Network
        nn_out = best_model.nn_amp(log_s_z, Delta_z)
        
        # Unpack the Neural Network output [500, 2]
        # Column 0 = Real, Column 1 = Imaginary
        res_R = nn_out[:, 0].cpu().numpy()
        res_I = nn_out[:, 1].cpu().numpy()

    label = f'{E/1000:g} TeV' if E >= 1000 else f'{E:g} GeV'
    
    # Calculate the modulated residual: Delta * NN / sqrt(s)
    # Note: sqrt(s) is exactly E. We use numpy arrays for plotting.
    y_plot_R = Delta_eval_np * res_R / E
    y_plot_I = Delta_eval_np * res_I / E
    
    ax1.plot(t_eval, y_plot_R, label=label, linewidth=2)
    ax2.plot(t_eval, y_plot_I, label=label, linewidth=2)

# -------------------------------------------------------------
# Formatting Real Axis
# -------------------------------------------------------------
ax1.axhline(0, color='black', lw=1, ls='--') 
ax1.set_title(r'Real Residual Correction ($\Delta \cdot \text{NN}_R / \sqrt{s}$)', fontsize=14)
ax1.set_xlabel(r'$|t| \ (\text{GeV}^2)$', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# -------------------------------------------------------------
# Formatting Imaginary Axis
# -------------------------------------------------------------
ax2.axhline(0, color='black', lw=1, ls='--') 
ax2.set_title(r'Imaginary Residual Correction ($\Delta \cdot \text{NN}_I / \sqrt{s}$)', fontsize=14)
ax2.set_xlabel(r'$|t| \ (\text{GeV}^2)$', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("nn_residuals_fixed.png", dpi=300)
plt.show()

# -------------------------------------------------------------
# CROSS SECTION GRAPHS VS ENERGY (Extrapolation)
# -------------------------------------------------------------

# Experimental literature data (Inelastic cross-section)
x_lit = np.array([204.25, 906.2, 1968.4, 7350, 12481, 5.746e+04, 1.0505e+05])
y_lit = np.array([41.489, 50.337, 57.5, 70.983, 73.23, 91.208, 105.253])

best_model.eval()

# 1. Define the energy range (sqrt_s in GeV)
# 10^1 to 10^6 GeV (10 GeV to 1 PeV)
sqrt_s_range = np.logspace(1, 6, 100)
stot_list, sel_list, sinel_list = [], [], []

print("Calculating integrated cross-sections (10 to 10^6 GeV)...")

with tc.no_grad():
    for sqs in sqrt_s_range:
        # Convert sqrt(s) to the required tensor s
        s_val = sqs**2
        s_tc = tc.tensor([s_val], dtype=tc.float32, device=best_model.device)
        
        # Call the sigmas method directly from the best_model
        # It returns (sigma_el, sigma_inel, sigma_tot, ratio)
        sig_el, sig_inel, sig_tot, _ = best_model.sigmas(s_tc, "pp")
        
        # Store as standard Python floats for matplotlib
        stot_list.append(sig_tot.item())
        sel_list.append(sig_el.item())
        sinel_list.append(sig_inel.item())

# -------------------------------------------------------------
# FINAL GRAPH PLOTTING
# -------------------------------------------------------------

plt.figure(figsize=(10, 7))

# Model Curves (Neural Network + Analytical Baseline)
plt.plot(sqrt_s_range, stot_list, 'k-',  lw=2, label=r'$\sigma_{tot}$ (Model)')
plt.plot(sqrt_s_range, sinel_list, 'r--', lw=2, label=r'$\sigma_{inel}$ (Model)')
plt.plot(sqrt_s_range, sel_list, 'b:',   lw=2, label=r'$\sigma_{el}$ (Model)')

# --- ADDING LITERATURE POINTS ---
plt.scatter(x_lit, y_lit, color='darkred', marker='s', s=40,
            label=r'$\sigma_{inel}$ (Literature)', zorder=5)

# Axis Configuration
plt.xscale('log')
plt.xlim(10, 1e6)
plt.ylim(10, 300)

plt.xlabel(r'$\sqrt{s}$ (GeV)', fontsize=14)
plt.ylabel(r'$\sigma$ (mb)', fontsize=14)
plt.title('Proton-Proton Cross Sections (Extrapolation)', fontsize=16)

# Elegant grid
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12, frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig("Integrated_cross_sections.png", dpi=300)
plt.show()