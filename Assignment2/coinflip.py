import numpy as np
import matplotlib.pyplot as plt

def flip_coins(num_coins, num_flips):
    # Simulate flipping num_coins coins num_flips times, with a fair coin (probability of heads = 0.5)
    flips = np.random.binomial(1, 0.5, (num_coins, num_flips))
    return flips

def experiment(num_coins=1000, num_flips=10, num_runs=100000):
    # Arrays to store the results for nu_1, nu_rand, and nu_min
    nu_1_values = []
    nu_rand_values = []
    nu_min_values = []
    
    for _ in range(num_runs):
        # Flip all coins
        flips = flip_coins(num_coins, num_flips)
        
        # Calculate the fraction of heads for each coin
        heads_fraction = np.mean(flips, axis=1)
        
        # Store nu_1 (first coin's heads fraction)
        nu_1 = heads_fraction[0]
        nu_1_values.append(nu_1)
        
        # Store nu_rand (randomly chosen coin's heads fraction)
        nu_rand = np.random.choice(heads_fraction)
        nu_rand_values.append(nu_rand)
        
        # Store nu_min (coin with minimum heads fraction)
        nu_min = np.min(heads_fraction)
        nu_min_values.append(nu_min)
    
    return np.array(nu_1_values), np.array(nu_rand_values), np.array(nu_min_values)

def plot_histograms(nu_1_values, nu_rand_values, nu_min_values):
    # Plot histograms of nu_1, nu_rand, and nu_min
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(nu_1_values, bins=10, alpha=0.7, color='blue')
    plt.title('nu_1 (First Coin)')
    plt.xlabel('Fraction of Heads')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(nu_rand_values, bins=10, alpha=0.7, color='green')
    plt.title('nu_rand (Random Coin)')
    plt.xlabel('Fraction of Heads')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(nu_min_values, bins=10, alpha=0.7, color='red')
    plt.title('nu_min (Min Heads Coin)')
    plt.xlabel('Fraction of Heads')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Run the experiment
nu_1, nu_rand, nu_min = experiment(num_coins=1000, num_flips=10, num_runs=100000)

# Plot the histograms
plot_histograms(nu_1, nu_rand, nu_min)
