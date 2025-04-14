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

    def forward_pass(self, inputs, weights=None):
        """Performs a forward pass through the network."""
        if weights is None:
            weights = self.weights
        inputs_with_bias = np.concatenate(([1], inputs))
        hidden_sum = np.dot(weights[0].T, inputs_with_bias)
        hidden_output = np.concatenate(([1], self.hidden_activation.compute(hidden_sum)))
        output_sum = np.dot(weights[1].T, hidden_output)
        final_output = self.output_activation.compute(output_sum)
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


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # PROBLEM 1
    # Instantiate helper classes
    data_processor = DataProcessor()
    gradient_tester = GradientTester()

    # Load and preprocess data
    train_set = data_processor.load_data('./ZipDigits.train')
    test_set = data_processor.load_data('./ZipDigits.test')
    combined_data = train_set + test_set

    symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in combined_data]
    intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in combined_data]

    norm_symmetry = data_processor.normalize_features(symmetry_data)
    norm_intensity = data_processor.normalize_features(intensity_data)

    train_data, test_data = data_processor.split_data(norm_intensity, norm_symmetry)

    # Part A: Testing network gradients
    print("Part A: Gradient Testing")
    gradient_tester.test_gradients(NeuralNetwork(TanhActivation(), IdentityActivation()), "Identity")
    gradient_tester.test_gradients(NeuralNetwork(TanhActivation(), TanhActivation()), "Tanh")

    # Part B: Gradient approximation with perturbation
    print("\nPart B: Gradient Approximation via Perturbation")
    gradient_tester.approximate_gradients((TanhActivation(), IdentityActivation()), "Identity")
    gradient_tester.approximate_gradients((TanhActivation(), TanhActivation()), "Tanh")

