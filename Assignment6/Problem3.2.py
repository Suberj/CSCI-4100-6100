import numpy as np
import matplotlib.pyplot as plt

# Function to generate points on a semi-circle
def generate_semi_circle(num_points, rad, thk, sep, label, upward=True, horizontal_shift=0):
    angles = np.pi * np.random.rand(num_points)  # random angles from 0 to pi
    radii = rad + thk * np.random.rand(num_points)  # random radii within the thickness
    x = radii * np.cos(angles) + horizontal_shift  # Apply horizontal shift for alignment
    if upward:
        y = radii * np.sin(angles) + sep  # Top semi-circle (curving upward)
    else:
        y = -radii * np.sin(angles) - sep  # Bottom semi-circle (curving downward)
    data = np.vstack((x, y)).T
    labels = np.ones(num_points) * label
    return data, labels

# Parameters
rad = 10
thk = 5
num_points = 5000  # Increase number of points per semi-circle for more stable results
num_trials = 5  # Number of trials to average over

# Horizontal shift to align the top semi-circle's center with the middle of the bottom semi-circle's edge
horizontal_shift = rad

# Perceptron Learning Algorithm (PLA) with maximum iterations
def pla(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])  # Initialize weights to zero
    num_iterations = 0
    while num_iterations < max_iter:
        misclassified_points = 0
        for i in range(X.shape[0]):
            if np.sign(np.dot(w, X[i])) != y[i]:  # If point is misclassified
                w = w + y[i] * X[i]  # Update the weights
                misclassified_points += 1
        if misclassified_points == 0:  # If no misclassified points, stop
            return num_iterations  # Return number of iterations
        num_iterations += 1
    return num_iterations  # If max iterations are reached, return it

# Function to run the experiment for different values of sep and plot the results
def run_experiment():
    sep_values = np.arange(0.2, 5.2, 0.2)  # Separation values from 0.2 to 5.0
    iterations_list = []

    for sep in sep_values:
        total_iterations = 0  # Track total iterations across trials

        for _ in range(num_trials):  # Run multiple trials for each sep
            # Generate points for the two semi-circles
            X_blue, y_blue = generate_semi_circle(num_points, rad, thk, sep, 1, upward=False)
            X_red, y_red = generate_semi_circle(num_points, rad, thk, sep, -1, upward=True, horizontal_shift=horizontal_shift)

            # Combine the two sets of points
            X = np.vstack((X_blue, X_red))
            y = np.hstack((y_blue, y_red))

            # Add a bias term to the data points (for the PLA)
            X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term as first column

            # Run PLA and accumulate the number of iterations
            iterations = pla(X_with_bias, y)
            total_iterations += iterations

        # Average iterations over the trials
        avg_iterations = total_iterations / num_trials
        iterations_list.append(avg_iterations)
        print(f"sep = {sep:.1f}, Average PLA iterations = {avg_iterations:.2f}")

    # Plot sep vs number of PLA iterations
    plt.figure(figsize=(8, 6))
    plt.plot(sep_values, iterations_list, marker='o', color='blue')
    plt.xlabel('Separation (sep)')
    plt.ylabel('Average Number of PLA Iterations')
    plt.title('Separation (sep) vs. Average Number of PLA Iterations')
    plt.grid(True)
    plt.show()

# Run the experiment
run_experiment()
