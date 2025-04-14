import numpy as np
import matplotlib.pyplot as plt

# Transformation function: X -> Z
def transform_to_z(x1, x2):
    z1 = x1 ** 3 - x2
    z2 = x1 * x2
    return np.array([z1, z2])

# Data points in X-space
x1_pos, x2_pos, y_pos = 1, 0, 1  # Positive class
x1_neg, x2_neg, y_neg = -1, 0, -1  # Negative class

# Transform data points to Z-space
z_pos = transform_to_z(x1_pos, x2_pos)
z_neg = transform_to_z(x1_neg, x2_neg)

# Optimal hyperplane in Z-space
z_mid = (z_pos + z_neg) / 2  # Midpoint
w_z = z_pos - z_neg         # Perpendicular vector to the line joining the points
b_z = -np.dot(w_z, z_mid)   # Bias term for the hyperplane

# Decision boundary in Z-space
def decision_boundary_z(z1):
    return -(w_z[0] * z1 + b_z) / w_z[1]

# Generate grid of X-space points
x1_vals = np.linspace(-1.5, 1.5, 200)
x2_vals = np.linspace(-1.5, 1.5, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
X1_flat = X1.ravel()
X2_flat = X2.ravel()

# Classify X-space points based on Z-space transformation
def classify_x(x1, x2):
    z = transform_to_z(x1, x2)
    return np.sign(np.dot(w_z, z) + b_z)

Z_X = np.array([classify_x(x1, x2) for x1, x2 in zip(X1_flat, X2_flat)])
Z_X = Z_X.reshape(X1.shape)

# Generate grid of Z-space points
z1_vals = np.linspace(-1.5, 1.5, 100)

# Plot combined decision boundaries
plt.figure(figsize=(8, 6))

# Plot Z-space boundary as a vertical line (optimal hyperplane in Z-space)
plt.axvline(x=0, color="blue", linestyle="-", linewidth=2, label="Z-space boundary")

# Plot decision boundary in X-space
plt.contour(X1, X2, Z_X, levels=[0], colors=["red"], linewidths=2, label="X-space boundary")

# Scatter original points in X-space
plt.scatter([x1_pos, x1_neg], [x2_pos, x2_neg], c=["blue", "red"], label="Original Points in X-space", s=100)

# Configure plot
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("SVM Decision Boundaries for Two Points")
plt.grid()
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.show()
