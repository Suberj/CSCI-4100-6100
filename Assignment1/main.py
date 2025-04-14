import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from perceptron_algorithm import perceptron_learning_algorithm

# Define the folder containing the datasets
folder = r"D:\Machine Learning\Assignment1"  # Use raw string to handle Windows file paths

# List of dataset filenames and corresponding target function parameters (slope, intercept)
datasets = [
    {"filename": "part_b_dataset.csv", "slope": -0.76, "intercept": -0.01},
    {"filename": "part_c_dataset.csv", "slope": -0.45, "intercept": 0.10},  # Example values, adjust if needed
    {"filename": "part_d_dataset.csv", "slope": -0.55, "intercept": 0.05},  # Example values
    {"filename": "part_e_dataset.csv", "slope": -0.60, "intercept": -0.10}  # Example values
]

# Loop through each dataset
for dataset in datasets:
    filename = dataset["filename"]
    target_slope = dataset["slope"]
    target_intercept = dataset["intercept"]
    
    # Load the dataset
    data = pd.read_csv(os.path.join(folder, filename))
    
    # Prepare the input (X) and output (y)
    X = data[["x1", "x2"]].values
    y = data["label"].values
    
    # Add bias term to the input data (for weights w0)
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Run the perceptron learning algorithm on the dataset
    weights, iterations = perceptron_learning_algorithm(X_bias, y)
    
    # Print the results for this dataset
    print(f"Dataset: {filename}")
    print(f"Number of iterations to converge: {iterations}")
    print(f"Learned weights: {weights}\n")
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    
    # Plot the target function based on known slope and intercept
    x_vals = np.linspace(-1.5, 1.5, 100)
    y_vals_target = target_slope * x_vals + target_intercept
    plt.plot(x_vals, y_vals_target, 'b-', label=f"Target Function: y = {target_slope:.2f}x + {target_intercept:.2f}")
    
    # Plot the final hypothesis learned by the perceptron
    final_slope = -weights[1] / weights[2]
    final_intercept = -weights[0] / weights[2]
    y_vals_hypothesis = final_slope * x_vals + final_intercept
    plt.plot(x_vals, y_vals_hypothesis, 'g--', label=f"Final Hypothesis: y = {final_slope:.2f}x + {final_intercept:.2f}")
    
    # Plot the data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', marker='o', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='Class -1')
    
    # Label the axes
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.title(f"Perceptron Learning Results for {filename} (Iterations: {iterations})")
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # Show the plot
    plt.show()
