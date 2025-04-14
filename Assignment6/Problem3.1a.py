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
sep = 5  # Vertical separation between the two semi-circles
num_points = 1000

# Horizontal shift to align the top semi-circle's center with the middle of the bottom semi-circle's edge
horizontal_shift = rad

# Generate points for the two semi-circles
X_blue, y_blue = generate_semi_circle(num_points, rad, thk, sep, 1, upward=False)  # Blue semi-circle (bottom, no shift)
X_red, y_red = generate_semi_circle(num_points, rad, thk, sep, -1, upward=True, horizontal_shift=horizontal_shift)  # Red semi-circle (top, shifted horizontally)

# Combine the two sets of points
X = np.vstack((X_blue, X_red))
y = np.hstack((y_blue, y_red))

# Add a bias term to the data points (for the PLA)
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term as first column

# Initialize weights to zero
w = np.zeros(X_with_bias.shape[1])

# Perceptron Learning Algorithm (PLA) with maximum iterations
def pla(X, y, w, max_iter=1000):
    num_iterations = 0
    while num_iterations < max_iter:
        misclassified_points = 0
        for i in range(X.shape[0]):
            if np.sign(np.dot(w, X[i])) != y[i]:  # If point is misclassified
                w = w + y[i] * X[i]  # Update the weights
                misclassified_points += 1
        if misclassified_points == 0:  # If no misclassified points, stop
            print(f"PLA converged in {num_iterations} iterations.")
            return w, num_iterations
        num_iterations += 1
    print(f"PLA reached the maximum number of iterations: {max_iter}")
    return w, num_iterations

# Run PLA with a limit on iterations
final_w, iterations = pla(X_with_bias, y, w)

# Plot the data and final decision boundary
plt.figure(figsize=(8, 8))  # Adjust figure size for better visibility

# Plot blue (+1) points
plt.scatter(X_blue[:, 0], X_blue[:, 1], color='blue', label='Class +1 (Blue)', alpha=0.6)

# Plot red (-1) points
plt.scatter(X_red[:, 0], X_red[:, 1], color='red', label='Class -1 (Red)', alpha=0.6)

# Plot the decision boundary (w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1) / w2)
x_vals = np.linspace(-21, 26, 100)
y_vals = -(final_w[0] + final_w[1] * x_vals) / final_w[2]
plt.plot(x_vals, y_vals, label='Decision Boundary', color='green')

# Display the final hypothesis line equation
w0, w1, w2 = final_w  # Extract weights
hypothesis_eq = f"Final Hypothesis: {w1:.2f}x1 + {w2:.2f}x2 + {w0:.2f} = 0"
plt.text(-18, 23, hypothesis_eq, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))  # Position the text

# Set plot labels and legend
plt.xlim([-21, 26])  # Adjust the x-axis limits to -21 to 26
plt.ylim([-21, 26])  # Adjust the y-axis limits to -21 to 26
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('PLA: Data and Final Hypothesis')
plt.legend()
plt.grid(True)
plt.show()
