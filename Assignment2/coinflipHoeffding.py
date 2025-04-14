import numpy as np
import matplotlib.pyplot as plt

# Function to simulate coin flips and compute the fractions of heads
def flip_coins(num_coins, num_flips):
    # Simulate flipping num_coins coins num_flips times
    flips = np.random.binomial(1, 0.5, (num_coins, num_flips))
    return flips

# Perform the experiment for multiple runs and collect nu values
def experiment(num_coins=1000, num_flips=10, num_runs=100000):
    nu_1_values = []
    nu_rand_values = []
    nu_min_values = []
    
    for _ in range(num_runs):
        # Flip all coins
        flips = flip_coins(num_coins, num_flips)
        
        # Calculate the fraction of heads for each coin
        heads_fraction = np.mean(flips, axis=1)
        
        # Store nu_1 (first coin's heads fraction)
        nu_1_values.append(heads_fraction[0])
        
        # Store nu_rand (randomly chosen coin's heads fraction)
        nu_rand_values.append(np.random.choice(heads_fraction))
        
        # Store nu_min (coin with minimum heads fraction)
        nu_min_values.append(np.min(heads_fraction))
    
    return np.array(nu_1_values), np.array(nu_rand_values), np.array(nu_min_values)

# Compute the Hoeffding bound
def hoeffding_bound(epsilon, N):
    return 2 * np.exp(-2 * N * epsilon**2)

# Calculate P(|nu - mu| >= epsilon) for different epsilon values
def compute_probabilities(nu_values, mu, epsilons):
    probabilities = []
    for epsilon in epsilons:
        prob = np.mean(np.abs(nu_values - mu) >= epsilon)
        probabilities.append(prob)
    return np.array(probabilities)

# Plotting function
def plot_hoeffding_and_empirical(epsilons, prob_nu1, prob_nurand, prob_numin, N):
    plt.figure(figsize=(10, 6))

    # Plot empirical probabilities
    plt.plot(epsilons, prob_nu1, label=r'Empirical $P(|\nu_1 - \mu| \geq \epsilon)$', marker='o', linestyle='-')
    plt.plot(epsilons, prob_nurand, label=r'Empirical $P(|\nu_{\text{rand}} - \mu| \geq \epsilon)$', marker='o', linestyle='-')
    plt.plot(epsilons, prob_numin, label=r'Empirical $P(|\nu_{\text{min}} - \mu| \geq \epsilon)$', marker='o', linestyle='-')

    # Plot Hoeffding bound
    hoeffding_bounds = hoeffding_bound(epsilons, N)
    plt.plot(epsilons, hoeffding_bounds, label=r'Hoeffding Bound', color='black', linestyle='--')

    # Customize plot
    plt.yscale('log')  # Use log scale for y-axis
    plt.xlabel(r'$\epsilon$ (Deviation from $\mu$)')
    plt.ylabel(r'Probability $P(|\nu - \mu| \geq \epsilon)$')
    plt.title('Hoeffding Bound vs Empirical Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
num_coins = 1000
num_flips = 10
num_runs = 100000
mu = 0.5
N = num_flips
epsilons = np.linspace(0.05, 0.5, 10)  # Epsilon values

# Run the experiment
nu_1_values, nu_rand_values, nu_min_values = experiment(num_coins, num_flips, num_runs)

# Compute probabilities for each epsilon
prob_nu1 = compute_probabilities(nu_1_values, mu, epsilons)
prob_nurand = compute_probabilities(nu_rand_values, mu, epsilons)
prob_numin = compute_probabilities(nu_min_values, mu, epsilons)

# Plot the results
plot_hoeffding_and_empirical(epsilons, prob_nu1, prob_nurand, prob_numin, N)
