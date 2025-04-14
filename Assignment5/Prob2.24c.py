import numpy as np
import matplotlib.pyplot as plt

# Number of experiments
n_experiments = 10000

# Function to fit a linear model to two points and return coefficients a and b
def fit_linear_model(x1, x2, y1, y2):
    a = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - a * x1            # intercept
    return a, b

# Function to compute the squared error between two functions
def squared_error(f1, f2, x_vals):
    return np.mean((f1(x_vals) - f2(x_vals)) ** 2)

# Target function f(x) = x^2
def f(x):
    return x ** 2

# Generate test points uniformly in [-1, 1] to evaluate errors
x_test = np.linspace(-1, 1, 100)

# Store coefficients for all experiments
a_values = []
b_values = []

# Run the experiment multiple times
for _ in range(n_experiments):
    # Randomly sample two points in the range [-1, 1]
    x1, x2 = np.random.uniform(-1, 1, 2)
    # Compute their corresponding target points
    y1, y2 = x1**2, x2**2
    # Fit a linear model to these two points
    a, b = fit_linear_model(x1, x2, y1, y2)
    a_values.append(a)
    b_values.append(b)

# Average coefficients for the average hypothesis
a_avg = np.mean(a_values)
b_avg = np.mean(b_values)

# Average hypothesis g_bar(x)
def g_bar(x):
    return a_avg * x + b_avg

# Compute E[E_out], bias, and variance
E_out_total = 0
bias_total = 0
variance_total = 0

for i in range(n_experiments):
    a_i = a_values[i]
    b_i = b_values[i]

    # Hypothesis for this experiment g_i(x)
    def g_i(x):
        return a_i * x + b_i
    
    # Compute E_out for this hypothesis
    E_out_total += squared_error(g_i, f, x_test)
    
    # Compute the bias
    bias_total += squared_error(g_bar, f, x_test)
    
    # Compute the variance for this hypothesis
    variance_total += squared_error(g_i, g_bar, x_test)

# Average values
E_out_avg = E_out_total / n_experiments
bias_avg = bias_total / n_experiments
variance_avg = variance_total / n_experiments

# Print the results
print(f"Expected E_out: {E_out_avg}")
print(f"Bias^2: {bias_avg}")
print(f"Variance: {variance_avg}")
print(f"Bias^2 + Variance: {bias_avg + variance_avg}")

# Plot the target function f(x) and average hypothesis g_bar(x)
plt.plot(x_test, f(x_test), label='f(x) = x^2', color='blue')
plt.plot(x_test, g_bar(x_test), label='g_bar(x) (Average hypothesis)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of Target Function and Average Hypothesis')
plt.show()
