import numpy as np
import matplotlib.pyplot as plt

# Example: Neural activity over time
np.random.seed(42)
activity = np.random.normal(0.1, 0.05, 1000)  # Mean 0.1, std 0.05

# First moment: MEAN (average)
mean = np.mean(activity)  # ≈ 0.1

# Second moment: VARIANCE (spread)
variance = np.var(activity)  # ≈ 0.0025

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution
ax = axes[0]
ax.hist(activity, bins=30, density=True, alpha=0.7, color='blue')
ax.axvline(mean, color='red', linewidth=2, label=f'Mean={mean:.3f}')
ax.axvline(mean - np.sqrt(variance), color='orange', linestyle='--', label='±1 std')
ax.axvline(mean + np.sqrt(variance), color='orange', linestyle='--')
ax.set_xlabel('Neural Activity')
ax.set_ylabel('Probability')
ax.set_title('Distribution of Activity')
ax.legend()

# Time series
ax = axes[1]
time = np.arange(len(activity))
ax.plot(time[:200], activity[:200], 'b-', alpha=0.7, linewidth=0.5)
ax.axhline(mean, color='red', linewidth=2, label='Mean')
ax.fill_between(time[:200], mean - np.sqrt(variance), mean + np.sqrt(variance),
                color='orange', alpha=0.3, label='±1 std')
ax.set_xlabel('Time')
ax.set_ylabel('Activity')
ax.set_title('Activity Over Time')
ax.legend()

# What variance tells us
ax = axes[2]
# Low variance example
low_var = np.random.normal(0.1, 0.01, 1000)
# High variance example  
high_var = np.random.normal(0.1, 0.1, 1000)
ax.hist(low_var, bins=30, alpha=0.5, color='green', label='Low variance', density=True)
ax.hist(high_var, bins=30, alpha=0.5, color='red', label='High variance', density=True)
ax.set_xlabel('Activity')
ax.set_ylabel('Probability')
ax.set_title('Same Mean, Different Variance')
ax.legend()

plt.tight_layout()
plt.show()

# Demonstration: Why means alone are insufficient
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Two populations with same mean, different variance
t = np.linspace(0, 100, 1000)
mean_activity = 0.1

# Population 1: Low variance (stable)
pop1 = mean_activity + 0.01 * np.random.randn(len(t))
# Population 2: High variance (fluctuating)
pop2 = mean_activity + 0.05 * np.random.randn(len(t))

# Plot time series
ax = axes[0, 0]
ax.plot(t, pop1, 'b-', alpha=0.7, label='Low variance')
ax.plot(t, pop2, 'r-', alpha=0.7, label='High variance')
ax.axhline(mean_activity, color='k', linestyle='--', label='Mean')
ax.set_xlabel('Time')
ax.set_ylabel('Activity')
ax.set_title('Same Mean, Different Dynamics')
ax.legend()

# Impact on downstream neurons
ax = axes[0, 1]
# Threshold for downstream activation
threshold = 0.15
# Count how often each population exceeds threshold
exceed1 = np.sum(pop1 > threshold) / len(pop1) * 100
exceed2 = np.sum(pop2 > threshold) / len(pop2) * 100
ax.bar(['Low var', 'High var'], [exceed1, exceed2], color=['blue', 'red'])
ax.set_ylabel('% Time Above Threshold')
ax.set_title(f'Threshold Crossing (θ={threshold})')

# Firing rate with threshold
ax = axes[1, 0]
thresholds = np.linspace(0, 0.3, 100)
from scipy.special import erf

# For Gaussian input through threshold
def firing_rate(mean, variance, threshold):
    z = (mean - threshold) / np.sqrt(2 * variance)
    return 0.5 * (1 + erf(z))

fr_low = [firing_rate(mean_activity, 0.01**2, th) for th in thresholds]
fr_high = [firing_rate(mean_activity, 0.05**2, th) for th in thresholds]

ax.plot(thresholds, fr_low, 'b-', linewidth=2, label='Low variance')
ax.plot(thresholds, fr_high, 'r-', linewidth=2, label='High variance')
ax.axvline(mean_activity, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Threshold')
ax.set_ylabel('Firing Rate')
ax.set_title('Variance Changes Input-Output Function')
ax.legend()

# Near criticality: variance explosion
ax = axes[1, 1]
rho_values = np.linspace(0.8, 1.05, 100)
# Variance scales as 1/(1-ρ)² near criticality
variance_scaling = 1 / (1 - rho_values)**2
variance_scaling[rho_values > 1] = 100  # Cap for visualization
ax.semilogy(rho_values, variance_scaling, 'g-', linewidth=3)
ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Critical point')
ax.set_xlabel('Spectral Radius ρ')
ax.set_ylabel('Variance Scaling')
ax.set_title('Variance Diverges at Criticality')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()