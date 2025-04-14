import random
import numpy as np
import matplotlib.pyplot as plt

# --- DATA PROCESSING UTILITIES ---
class DataProcessor:
    """Class for data loading and feature extraction."""
    
    def __init__(self):
        pass  # No instance variables needed

    def load_data(self, filepath):
        """Reads and processes the dataset from the given file."""
        dataset = []
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 256:  # Skip incomplete rows
                    continue
                label = int(float(parts[0]))
                pixels = np.array([float(p) for p in parts[1:]]).reshape(16, -1)
                dataset.append((1 if label == 1 else -1, pixels))
        return dataset

    def calculate_symmetry(self, image):
        """Computes symmetry for a given 16x16 image."""
        img = image.reshape(16, 16)
        left_half = img[:, :8]
        right_half = np.flip(img[:, 8:], axis=1)
        return 1 - np.sum(np.abs(left_half - right_half)) / np.sum(np.abs(img))

    def calculate_intensity(self, image):
        """Calculates intensity for the given image."""
        return np.mean(image)

    def normalize_features(self, data, target_min=-1.0, target_max=1.0):
        """Normalizes feature data between the specified target range."""
        features = [value for _, value in data]
        orig_min, orig_max = min(features), max(features)
        scale = (target_max - target_min) / (orig_max - orig_min)
        shift = (target_min + target_max) / 2 - (orig_max + orig_min) / 2
        normalize = lambda val: (val + shift) * scale
        return [(label, normalize(value)) for label, value in data]

    def split_data(self, *datasets, train_size=300):
        """Splits datasets into training and testing sets."""
        combined = []
        for idx in range(len(datasets[0])):
            features = tuple(ds[idx][1] for ds in datasets)
            combined.append((datasets[0][idx][0], features))
        random.shuffle(combined)
        return combined[:train_size], combined[train_size:]


# --- DATA VISUALIZATION ---
class DataVisualizer:
    """Class for visualizing datasets."""
    
    def __init__(self):
        pass  # No instance variables needed

    def plot_features(self, *datasets):
        """Plots the data using scatter plots."""
        fig, ax = plt.subplots()
        positives = []
        negatives = []
        for dataset in datasets:
            for label, feature in dataset:
                if label == 1:
                    positives.append(feature)
                else:
                    negatives.append(feature)
        ax.scatter(*zip(*positives), c='blue', marker='o', label="Positive (1)")
        ax.scatter(*zip(*negatives), c='red', marker='x', label="Negative (-1)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Symmetry")
        return fig, ax


# --- ACTIVATION FUNCTIONS ---
class ActivationFunction:
    """Base class for activation functions."""
    
    def compute(self, x):
        raise NotImplementedError("compute() not implemented!")
    
    def derivative(self, x):
        raise NotImplementedError("derivative() not implemented!")

class TanhActivation(ActivationFunction):
    """Hyperbolic tangent activation function."""
    
    def compute(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - x**2

class IdentityActivation(ActivationFunction):
    """Identity activation function."""
    
    def compute(self, x):
        return x

    def derivative(self, x):
        return 1


# --- NEURAL NETWORK IMPLEMENTATION ---
class NeuralNetwork:
    """Simple neural network with one hidden layer."""
    
    def __init__(self, hidden_activation, output_activation, hidden_neurons=2, weight_init=0.25):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.weights = [
            np.full((3, hidden_neurons), weight_init),
            np.full((hidden_neurons + 1, 1), weight_init)
        ]

    """def forward_pass(self, inputs, weights=None):
        if weights is None:
            weights = self.weights
        inputs_with_bias = np.concatenate(([1], inputs))
        hidden_sum = np.dot(weights[0].T, inputs_with_bias)
        hidden_output = np.concatenate(([1], self.hidden_activation.compute(hidden_sum)))
        output_sum = np.dot(weights[1].T, hidden_output)
        final_output = self.output_activation.compute(output_sum)
        return inputs_with_bias, hidden_output, final_output"""
    
    def forward_pass(self, inputs, weights=None):
        """Performs a forward pass through the network. Supports batch processing."""
        if weights is None:
            weights = self.weights
        if inputs.ndim == 1:  # Single data point
            inputs = np.expand_dims(inputs, axis=0)  # Convert to batch of size 1
        inputs_with_bias = np.c_[np.ones((inputs.shape[0], 1)), inputs]  # Add bias
        hidden_sum = np.dot(inputs_with_bias, weights[0])  # Input to hidden layer
        hidden_output = np.c_[np.ones((inputs.shape[0], 1)), self.hidden_activation.compute(hidden_sum)]  # Hidden outputs
        output_sum = np.dot(hidden_output, weights[1])  # Input to output layer
        final_output = self.output_activation.compute(output_sum)  # Final outputs
        return inputs_with_bias, hidden_output, final_output

    def backward_pass(self, forward_cache, target, weights=None):
        """Performs a backward pass to calculate errors."""
        if weights is None:
            weights = self.weights
        inputs, hidden_outputs, final_output = forward_cache
        output_error = 2 * (final_output - target) * self.output_activation.derivative(final_output)
        hidden_error = self.hidden_activation.derivative(hidden_outputs[1:]) * np.dot(weights[1][1:], output_error)
        return hidden_error, output_error

    def compute_gradients(self, forward_cache, deltas):
        """Computes gradients for weight updates."""
        inputs, hidden_outputs, _ = forward_cache
        input_grad = np.outer(inputs, deltas[0]) / 4
        hidden_grad = np.outer(hidden_outputs, deltas[1]) / 4
        return input_grad, hidden_grad


# --- GRADIENT TESTING AND APPROXIMATION ---
class GradientTester:
    """Class for testing gradients and approximating them."""
    
    def __init__(self):
        pass  # No instance variables needed

    def test_gradients(self, nn, description):
        """Tests computed gradients using backpropagation."""
        cache = nn.forward_pass(np.array([1, 2]))
        deltas = nn.backward_pass(cache, 1)
        grads = nn.compute_gradients(cache, deltas)
        print(f"{description}:")
        print(f"Layer 1 Gradients: {grads[0]}")
        print(f"Layer 2 Gradients: {grads[1]}")

    def approximate_gradients(self, activations, desc):
        """Approximates gradients using finite differences."""
        hidden_act, output_act = activations
        nn1 = NeuralNetwork(hidden_act, output_act, weight_init=0.25)
        nn2 = NeuralNetwork(hidden_act, output_act, weight_init=0.25)
        gradients = [np.zeros_like(layer) for layer in nn1.weights]

        for l in range(len(nn1.weights)):
            for i in range(nn1.weights[l].shape[0]):
                for j in range(nn1.weights[l].shape[1]):
                    # Perturb the weight
                    nn1.weights[l][i, j] -= 0.0001
                    nn2.weights[l][i, j] += 0.0001

                    # Compute the outputs and errors
                    e1 = (nn1.forward_pass(np.array([1, 2]))[2] - 1) ** 2 / 4
                    e2 = (nn2.forward_pass(np.array([1, 2]))[2] - 1) ** 2 / 4

                    # Ensure e1 and e2 are scalars to avoid deprecation warnings
                    e1 = e1.item() if isinstance(e1, np.ndarray) else e1
                    e2 = e2.item() if isinstance(e2, np.ndarray) else e2

                    # Calculate gradient
                    gradients[l][i, j] = (e2 - e1) / 0.0002

                    # Reset the weights to their original values
                    nn1.weights[l][i, j] = 0.25
                    nn2.weights[l][i, j] = 0.25

        # Print the results
        print(f"{desc}:")
        print(f"Layer 1 Gradient: {gradients[0]}")
        print(f"Layer 2 Gradient: {gradients[1]}")
        
# --- PROBLEM 2A FUNCTION ---
def problem_2a(train_data, nn, learning_rate=0.1, max_iterations=2_000_000, batch_size=300):
    """
    Optimized training for Problem 2a using batch gradient descent with a configurable batch size.
    Includes decision boundary visualization after training.
    """
    errors = []
    num_features = len(train_data[0][1])
    batch_size = min(batch_size, len(train_data))  # Ensure batch size is not larger than the dataset

    for iteration in range(max_iterations):
        # Sample a random batch of data
        batch = random.sample(train_data, batch_size)
        batch_features = np.array([features for _, features in batch])
        batch_labels = np.array([label for label, _ in batch])

        # Forward pass for the batch
        batch_inputs = np.c_[np.ones(batch_size), batch_features]  # Add bias term
        hidden_sums = np.dot(batch_inputs, nn.weights[0])  # Input to hidden layer
        hidden_outputs = np.c_[np.ones(batch_size), nn.hidden_activation.compute(hidden_sums)]  # Hidden layer outputs
        output_sums = np.dot(hidden_outputs, nn.weights[1])  # Input to output layer
        outputs = nn.output_activation.compute(output_sums)  # Final outputs

        # Compute error
        errors_batch = (outputs.squeeze() - batch_labels) ** 2 / 4
        error_mean = np.mean(errors_batch)

        # Backward pass
        output_deltas = 2 * (outputs.squeeze() - batch_labels) * nn.output_activation.derivative(outputs)
        output_deltas = output_deltas[:, None]  # Reshape to (batch_size, 1) for correct matrix multiplication
        hidden_deltas = nn.hidden_activation.derivative(hidden_outputs[:, 1:]) * np.dot(output_deltas, nn.weights[1][1:].T)

        # Compute gradients
        grad_w1 = np.dot(batch_inputs.T, hidden_deltas) / batch_size
        grad_w2 = np.dot(hidden_outputs.T, output_deltas) / batch_size

        # Update weights
        nn.weights[0] -= learning_rate * grad_w1
        nn.weights[1] -= learning_rate * grad_w2

        # Log error periodically
        if iteration % 100 == 0:
            errors.append(error_mean)

    # Plot error
    plt.figure()
    plt.plot(range(0, max_iterations, 100), errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Problem 2a: $E_{in}$ vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('$E_{in}$')
    plt.show()

    # Decision boundary visualization
    fig, ax = plt.subplots()
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Symmetry')

    # Scatter the training data
    positives = [features for label, features in train_data if label == 1]
    negatives = [features for label, features in train_data if label == -1]
    ax.scatter(*zip(*positives), c='blue', label='Positive (1)', alpha=0.6)
    ax.scatter(*zip(*negatives), c='red', label='Negative (-1)', alpha=0.6)

    # Generate decision boundary
    x_vals = np.linspace(-1, 1, 200)
    y_vals = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            _, _, output = nn.forward_pass([X[i, j], Y[i, j]])
            Z[i, j] = np.sign(output)  # Classification boundary

    ax.contourf(X, Y, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
    plt.legend()
    plt.show()

# --- PROBLEM 2B FUNCTION ---
def problem_2b(train_data, nn, learning_rate=0.01, max_iterations=20_000_000, batch_size=300):
    """
    Optimized training for Problem 2b using mini-batch SGD.
    Includes adaptive learning rates, early stopping, and decision boundary visualization.
    """
    errors = []
    num_samples = len(train_data)
    batch_size = min(batch_size, num_samples)  # Ensure batch size does not exceed dataset size

    # Convert data to NumPy arrays for efficiency
    train_features = np.array([features for _, features in train_data])
    train_labels = np.array([label for label, _ in train_data])

    for iteration in range(max_iterations):
        # Sample a random batch of data
        batch_indices = random.sample(range(num_samples), batch_size)
        batch_features = train_features[batch_indices]
        batch_labels = train_labels[batch_indices]

        # Forward pass for the batch
        batch_inputs = np.c_[np.ones(batch_size), batch_features]  # Add bias term
        hidden_sums = np.dot(batch_inputs, nn.weights[0])  # Input to hidden layer
        hidden_outputs = np.c_[np.ones(batch_size), nn.hidden_activation.compute(hidden_sums)]  # Hidden layer outputs
        output_sums = np.dot(hidden_outputs, nn.weights[1])  # Input to output layer
        outputs = nn.output_activation.compute(output_sums)  # Final outputs

        # Compute error
        errors_batch = (outputs.squeeze() - batch_labels) ** 2 / 4
        error_mean = np.mean(errors_batch)

        # Backward pass
        output_deltas = 2 * (outputs.squeeze() - batch_labels) * nn.output_activation.derivative(outputs)
        output_deltas = output_deltas[:, None]  # Reshape to (batch_size, 1) for correct matrix multiplication
        hidden_deltas = nn.hidden_activation.derivative(hidden_outputs[:, 1:]) * np.dot(output_deltas, nn.weights[1][1:].T)

        # Compute gradients
        grad_w1 = np.dot(batch_inputs.T, hidden_deltas) / batch_size
        grad_w2 = np.dot(hidden_outputs.T, output_deltas) / batch_size

        # Update weights with adaptive learning rate
        adaptive_lr = learning_rate * (0.99 ** (iteration // 1000))  # Decay learning rate every 1000 iterations
        nn.weights[0] -= adaptive_lr * grad_w1
        nn.weights[1] -= adaptive_lr * grad_w2

        # Log error periodically
        if iteration % (num_samples // 10) == 0:  # Log every 10% of an epoch
            errors.append(error_mean)

    # Plot error
    plt.figure()
    x_values = [i * (num_samples // 10) for i in range(len(errors))]  # Match x-values to logged errors
    plt.plot(x_values, errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Problem 2b: $E_{in}$ vs Iteration # / N')
    plt.xlabel('Iteration # / N')
    plt.ylabel('$E_{in}$')
    plt.show()

    # Decision boundary visualization
    fig, ax = plt.subplots()
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Symmetry')

    # Scatter the training data
    positives = train_features[train_labels == 1]
    negatives = train_features[train_labels == -1]
    ax.scatter(positives[:, 0], positives[:, 1], c='blue', label='Positive (1)', alpha=0.6)
    ax.scatter(negatives[:, 0], negatives[:, 1], c='red', label='Negative (-1)', alpha=0.6)

    # Generate decision boundary
    x_vals = np.linspace(-1, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[X.ravel(), Y.ravel()]  # Flatten the grid into a batch
    _, _, outputs = nn.forward_pass(grid_points)  # Forward pass for all grid points
    Z = np.sign(outputs).reshape(X.shape)  # Reshape outputs back to grid shape

    ax.contourf(X, Y, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
    plt.legend()
    plt.show()

# --- PROBLEM 2C FUNCTION ---    
def problem_2c(train_data, nn, learning_rate=0.01, max_iterations=200_000, batch_size=300):
    """
    Optimized training for Problem 2c using mini-batch SGD with weight decay.
    """
    errors = []
    num_samples = len(train_data)
    batch_size = min(batch_size, num_samples)  # Ensure batch size does not exceed dataset size
    lambda_weight_decay = 0.01 / num_samples  # Weight decay parameter

    # Convert data to NumPy arrays for efficiency
    train_features = np.array([features for _, features in train_data])
    train_labels = np.array([label for label, _ in train_data])

    for iteration in range(max_iterations):
        # Sample a random batch of data
        batch_indices = random.sample(range(num_samples), batch_size)
        batch_features = train_features[batch_indices]
        batch_labels = train_labels[batch_indices]

        # Forward pass for the batch
        batch_inputs = np.c_[np.ones(batch_size), batch_features]  # Add bias term
        hidden_sums = np.dot(batch_inputs, nn.weights[0])  # Input to hidden layer
        hidden_outputs = np.c_[np.ones(batch_size), nn.hidden_activation.compute(hidden_sums)]  # Hidden layer outputs
        output_sums = np.dot(hidden_outputs, nn.weights[1])  # Input to output layer
        outputs = nn.output_activation.compute(output_sums)  # Final outputs

        # Compute error (in-sample + weight decay)
        errors_batch = (outputs.squeeze() - batch_labels) ** 2 / 4
        weight_decay = lambda_weight_decay * (np.sum(nn.weights[0] ** 2) + np.sum(nn.weights[1] ** 2))
        error_mean = np.mean(errors_batch) + weight_decay

        # Backward pass
        output_deltas = 2 * (outputs.squeeze() - batch_labels) * nn.output_activation.derivative(outputs)
        output_deltas = output_deltas[:, None]  # Reshape to (batch_size, 1) for correct matrix multiplication
        hidden_deltas = nn.hidden_activation.derivative(hidden_outputs[:, 1:]) * np.dot(output_deltas, nn.weights[1][1:].T)

        # Compute gradients
        grad_w1 = np.dot(batch_inputs.T, hidden_deltas) / batch_size + 2 * lambda_weight_decay * nn.weights[0]
        grad_w2 = np.dot(hidden_outputs.T, output_deltas) / batch_size + 2 * lambda_weight_decay * nn.weights[1]

        # Update weights with learning rate
        nn.weights[0] -= learning_rate * grad_w1
        nn.weights[1] -= learning_rate * grad_w2

        # Log error periodically
        if iteration % (num_samples // 10) == 0:  # Log every 10% of an epoch
            errors.append(error_mean)

    # Plot error
    plt.figure()
    x_values = [i * (num_samples // 10) for i in range(len(errors))]  # Match x-values to logged errors
    plt.plot(x_values, errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Problem 2c: $E_{augmented}$ vs Iteration # / N')
    plt.xlabel('Iteration # / N')
    plt.ylabel('$E_{augmented}$')
    plt.show()

    # Decision boundary visualization
    fig, ax = plt.subplots()
    ax.set_title('Decision Boundary (With Weight Decay)')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Symmetry')

    # Scatter the training data
    positives = train_features[train_labels == 1]
    negatives = train_features[train_labels == -1]
    ax.scatter(positives[:, 0], positives[:, 1], c='blue', label='Positive (1)', alpha=0.6)
    ax.scatter(negatives[:, 0], negatives[:, 1], c='red', label='Negative (-1)', alpha=0.6)

    # Generate decision boundary
    x_vals = np.linspace(-1, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[X.ravel(), Y.ravel()]  # Flatten the grid into a batch
    _, _, outputs = nn.forward_pass(grid_points)  # Forward pass for all grid points
    Z = np.sign(outputs).reshape(X.shape)  # Reshape outputs back to grid shape

    ax.contourf(X, Y, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
    plt.legend()
    plt.show()
    
# --- PROBLEM 2D FUNCTION ---     
def problem_2d(train_data, nn, learning_rate=0.01, max_iterations=200_000, batch_size=300, validation_size=50):
    """
    Optimized training for Problem 2d using mini-batch SGD with early stopping.
    """
    # Split data into training and validation sets
    validation_data = train_data[:validation_size]
    training_data = train_data[validation_size:]
    num_samples = len(training_data)

    # Convert training and validation data to NumPy arrays
    train_features = np.array([features for _, features in training_data])
    train_labels = np.array([label for label, _ in training_data])
    val_features = np.array([features for _, features in validation_data])
    val_labels = np.array([label for label, _ in validation_data])

    errors = []
    validation_errors = []
    best_weights = None
    best_validation_error = float('inf')

    for iteration in range(max_iterations):
        # Ensure batch size does not exceed the training set size
        batch_size = min(batch_size, num_samples)
        batch_indices = random.sample(range(num_samples), batch_size)
        batch_features = train_features[batch_indices]
        batch_labels = train_labels[batch_indices]

        # Forward pass for the batch
        batch_inputs = np.c_[np.ones(batch_size), batch_features]  # Add bias term
        hidden_sums = np.dot(batch_inputs, nn.weights[0])  # Input to hidden layer
        hidden_outputs = np.c_[np.ones(batch_size), nn.hidden_activation.compute(hidden_sums)]  # Hidden layer outputs
        output_sums = np.dot(hidden_outputs, nn.weights[1])  # Input to output layer
        outputs = nn.output_activation.compute(output_sums)  # Final outputs

        # Compute in-sample error
        errors_batch = (outputs.squeeze() - batch_labels) ** 2 / 4
        error_mean = np.mean(errors_batch)

        # Backward pass
        output_deltas = 2 * (outputs.squeeze() - batch_labels) * nn.output_activation.derivative(outputs)
        output_deltas = output_deltas[:, None]  # Reshape to (batch_size, 1) for correct matrix multiplication
        hidden_deltas = nn.hidden_activation.derivative(hidden_outputs[:, 1:]) * np.dot(output_deltas, nn.weights[1][1:].T)

        # Compute gradients
        grad_w1 = np.dot(batch_inputs.T, hidden_deltas) / batch_size
        grad_w2 = np.dot(hidden_outputs.T, output_deltas) / batch_size

        # Update weights
        nn.weights[0] -= learning_rate * grad_w1
        nn.weights[1] -= learning_rate * grad_w2

        # Log in-sample error periodically
        if iteration % (num_samples // 10) == 0:  # Log every 10% of an epoch
            errors.append(error_mean)

            # Compute validation error
            val_inputs = np.c_[np.ones(len(val_features)), val_features]  # Add bias term
            val_hidden_sums = np.dot(val_inputs, nn.weights[0])  # Input to hidden layer
            val_hidden_outputs = np.c_[np.ones(len(val_features)), nn.hidden_activation.compute(val_hidden_sums)]  # Hidden outputs
            val_output_sums = np.dot(val_hidden_outputs, nn.weights[1])  # Input to output layer
            val_outputs = nn.output_activation.compute(val_output_sums).squeeze()  # Final outputs

            val_error = np.mean((val_outputs - val_labels) ** 2 / 4)
            validation_errors.append(val_error)

            # Early stopping: Save weights with minimum validation error
            if val_error < best_validation_error:
                best_validation_error = val_error
                best_weights = [nn.weights[0].copy(), nn.weights[1].copy()]

    # Use best weights for decision boundary visualization
    nn.weights = best_weights

    # Plot in-sample error
    plt.figure()
    x_values = [i * (num_samples // 10) for i in range(len(errors))]  # Match x-values to logged errors
    plt.plot(x_values, errors, label="$E_{in}$")
    plt.plot(x_values, validation_errors, label="$E_{val}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Problem 2d: $E_{in}$ and $E_{val}$ vs Iteration # / N')
    plt.xlabel('Iteration # / N')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # Decision boundary visualization
    fig, ax = plt.subplots()
    ax.set_title('Decision Boundary (With Early Stopping)')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Symmetry')

    # Scatter the training data
    positives = train_features[train_labels == 1]
    negatives = train_features[train_labels == -1]
    ax.scatter(positives[:, 0], positives[:, 1], c='blue', label='Positive (1)', alpha=0.6)
    ax.scatter(negatives[:, 0], negatives[:, 1], c='red', label='Negative (-1)', alpha=0.6)

    # Generate decision boundary
    x_vals = np.linspace(-1, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[X.ravel(), Y.ravel()]  # Flatten the grid into a batch
    _, _, outputs = nn.forward_pass(grid_points)  # Forward pass for all grid points
    Z = np.sign(outputs).reshape(X.shape)  # Reshape outputs back to grid shape

    ax.contourf(X, Y, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
    plt.legend()
    plt.show()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # Instantiate utilities
    data_processor = DataProcessor()
    data_visualizer = DataVisualizer()
    
    # Load data
    train_set = data_processor.load_data('./ZipDigits.train')
    test_set = data_processor.load_data('./ZipDigits.test')
    combined_data = train_set + test_set
    
    # Extract features
    symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in combined_data]
    intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in combined_data]
    norm_symmetry = data_processor.normalize_features(symmetry_data)
    norm_intensity = data_processor.normalize_features(intensity_data)
    train_data, test_data = data_processor.split_data(norm_intensity, norm_symmetry)
    
    # Neural network
    nn = NeuralNetwork(TanhActivation(), IdentityActivation(), hidden_neurons=10, weight_init=0.01)
    
    # Uncomment to run specific parts
    #problem_2a(train_data, nn)
    #problem_2b(train_data, nn)
    #problem_2c(train_data, nn)
    #problem_2d(train_data, nn)
    