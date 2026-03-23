# PINN-ElasticScattering ⚛️🤖

A research-grade implementation of **Physics-Informed Neural Networks (PINNs)** designed to reconstruct the complex scattering amplitudes of high-energy hadronic collisions ($pp$ and $\bar{p}p$).

An implementation of Physically Informed Neural Networks (PINNs) to extract the elastic scattering amplitude in $pp$ and $\bar{p}p$ collisions. The model uses Regge's theoretical foundation and COMPETE parameterizations as a baseline, optimizing residuals via replica Monte Carlo.

This project bridges the gap between traditional Regge Theory and Modern Machine Learning by using Neural Networks to correct analytical baselines while strictly obeying physical constraints such as **Unitarity**, **Analyticity**, and **Asymmetric Experimental Errors**.

## 📂 Project Structure

- **`neural.py`**: The training engine. It handles:
  - Data loading and normalization (Z-score).
  - Monte Carlo Replica generation with Gaussian fluctuations.
  - PINN training using a multi-objective Loss Function (Statistical $\chi^2$ + Curvature + Unitarity).
  - Advanced optimization logic including "Learning Rate Jumps" and Early Stopping.
- **`amplitude.py`**: The ensemble and analysis script. It:
  - Aggregates all trained replica models.
  - Computes the statistical mean and standard deviation (error bands).
  - Generates high-quality physics plots for $d\sigma/dt$, $\sigma_{tot}$, and $\rho(s)$.
- **`data.csv`**: Experimental dataset containing $\sqrt{s}$, $t$, $d\sigma/dt$, and mode ($pp$ or $\bar{p}p$).
- **`Artigos/`**: Bibliography and theoretical references supporting the physics implementation.
- **`venv/`**: Python virtual environment for dependency isolation.

---

## 🛠️ Requirements & Installation

The project is built using **PyTorch** for tensor computation and automatic differentiation. To set up the environment, ensure you have Python 3.8+ installed.

### Dependencies
- `torch`: Core Neural Network framework.
- `numpy`: Numerical operations and mask handling.
- `pandas`: Data manipulation and CSV parsing.
- `matplotlib`: Scientific visualization.

### Quick Start
```bash
# Clone the repository
git clone [https://github.com/joaovitorcl1000/PINN-ElasticScattering.git](https://github.com/joaovitorcl1000/PINN-ElasticScattering.git)
cd PINN-ElasticScattering
```

Create and configure the virtual environment. In the Linux (Ubuntu) terminal, run the following commands to isolate dependencies:

```bash
# Install virtual environment support (if not already installed)
sudo apt install python3-virtualenv

# Create the venv
virtualenv venv

# Activate the venv
source venv/bin/activate

# Update the package manager
python -m pip install --upgrade pip

# Install the necessary libraries
pip install numpy matplotlib pandas torch

# Run
python neural.py
```
