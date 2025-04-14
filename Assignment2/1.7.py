import numpy as np
import matplotlib.pyplot as plt

# Define parameters for N, mu, and the range of epsilon
N = 6
mu = 0.5
epsilon_values = np.linspace(0, 1, 100)

# Function to compute Hoeffding bound
def hoeffding_bound(epsilon, N):
    return 2 * np.exp(-2 * N * epsilon**2)

# Function to compute the probability that max deviation exceeds epsilon for 2 independent coins
def max_deviation_prob(epsilon, N):
    # P(|nu - mu| > epsilon) for a single coin
    P_single = hoeffding_bound(epsilon, N)
    # P(max_i |nu_i - mu_i| > epsilon) for two independent coins
    return 2 * P_single - P_single**2

# Calculate the probabilities for the plot
hoeffding_probs = hoeffding_bound(epsilon_values, N)
max_deviation_probs = max_deviation_prob(epsilon_values, N)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, max_deviation_probs, label=r'$P\left(\max_i |\nu_i - \mu_i| > \epsilon\right)$', color='blue')
plt.plot(epsilon_values, hoeffding_probs, label='Hoeffding Bound for a Single Coin', color='red', linestyle='--')

# Customize plot
plt.xlabel(r'$\epsilon$')
plt.ylabel('Probability')
plt.title('Maximum Deviation Probability vs Hoeffding Bound')
plt.legend()
plt.grid(True)
plt.show()
