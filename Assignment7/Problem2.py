import matplotlib.pyplot as plt
import numpy as np

# Cost function we want to minimize
def calculate_cost(x, y):
    return x**2 + 2 * y**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Update rules for x and y based on gradient descent
def update_position(x, y, learning_rate):
    grad_x = 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    grad_y = 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    x -= learning_rate * grad_x
    y -= learning_rate * grad_y
    return x, y

# Run gradient descent and collect cost values for plotting
def execute_gradient_descent(start_x=0.1, start_y=0.1, learning_rate=0.01, steps=50):
    history_x, history_y, costs = [], [], []
    x, y = start_x, start_y
    for _ in range(steps):
        x, y = update_position(x, y, learning_rate)
        costs.append(calculate_cost(x, y))
        history_x.append(x)
        history_y.append(y)
    return range(steps), costs, history_x, history_y

# Generate and display a line plot of the gradient descent process
def create_plot(iterations, costs, title):
    plt.figure()
    plt.plot(iterations, costs, marker='o')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

# Execute gradient descent for different learning rates and display plots
iter_steps_01, cost_values_01, _, _ = execute_gradient_descent(learning_rate=0.01)
create_plot(iter_steps_01, cost_values_01, "Gradient Descent: Learning Rate = 0.01")

iter_steps_1, cost_values_1, _, _ = execute_gradient_descent(learning_rate=0.1)
create_plot(iter_steps_1, cost_values_1, "Gradient Descent: Learning Rate = 0.1")

# Display summary of results for various starting points and learning rates
start_points = [(0.1, 0.1), (1, 1), (-0.5, -0.5), (-1, -1)]
learning_rates = [0.1, 0.01]

print("{:<20} {:<15} {:<15} {:<15} {:<15}".format("Starting Point", "Learning Rate", "Final x", "Final y", "Minimum Cost"))

for x_start, y_start in start_points:
    for rate in learning_rates:
        _, costs, x_path, y_path = execute_gradient_descent(x_start, y_start, rate)
        min_cost = min(costs)
        min_index = costs.index(min_cost)
        final_x, final_y = x_path[min_index], y_path[min_index]
        
        # Output the results in a formatted table style
        print("{:<20} {:<15.2f} {:<15.4f} {:<15.4f} {:<15.4f}".format(f"{(x_start, y_start)}", rate, final_x, final_y, min_cost))
