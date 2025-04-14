import numpy as np
import matplotlib.pyplot as plt
import random

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

# Polynomial kernel function
def polynomial_kernel(x1, x2, degree=8, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

# Manually implement SVM
class ManualSVM:
    def __init__(self, C=1.0, kernel=polynomial_kernel):
        self.C = C
        self.kernel = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = y
        self.kernel_matrix = np.array([[self.kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
        
        for _ in range(1000):  # Simple iterative optimization
            for i in range(n_samples):
                gradient = 1 - y[i] * (np.sum(self.alpha * y * self.kernel_matrix[i]) + self.b)
                if self.alpha[i] < self.C:
                    self.alpha[i] += 0.01 * gradient
                    self.alpha[i] = max(0, min(self.C, self.alpha[i]))  # Ensure 0 <= alpha <= C

        # Compute bias term
        self.b = np.mean([y[i] - np.sum(self.alpha * y * self.kernel_matrix[i]) for i in range(n_samples)])

    def predict(self, X):
        predictions = []
        for x in X:
            result = np.sum(self.alpha * self.y * np.array([self.kernel(x, sv) for sv in self.X])) + self.b
            predictions.append(np.sign(result))
        return np.array(predictions)

# Plot decision boundary
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['red', 'blue'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.6)
    plt.title(title)
    plt.xlabel('Feature 1 (Intensity)')
    plt.ylabel('Feature 2 (Symmetry)')
    plt.show()

# Cross-validation to find optimal C
def find_optimal_C(X, y, C_values, kernel):
    results = []
    for C in C_values:
        model = ManualSVM(C=C, kernel=kernel)
        model.fit(X, y)
        predictions = model.predict(X)
        E_in = np.mean(predictions != y)
        results.append((C, E_in))
        print(f"C = {C}, E_in = {E_in:.4f}")
    
    # Find the C value with the minimum E_in
    optimal_C, min_E_in = min(results, key=lambda x: x[1])
    print(f"Optimal C: {optimal_C}, Minimum E_in: {min_E_in:.4f}")
    return optimal_C

# Part (a): Train and display decision boundaries for small and large C
def part_a(data_processor, train_set, test_set):
    # Randomly sample 300 points for training
    sampled_data = random.sample(train_set, 300)

    # Extract features (intensity and symmetry)
    intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in sampled_data]
    symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in sampled_data]
    norm_intensity = data_processor.normalize_features(intensity_data)
    norm_symmetry = data_processor.normalize_features(symmetry_data)

    # Combine features for SVM
    X = np.array([[intensity, symmetry] for (_, intensity), (_, symmetry) in zip(norm_intensity, norm_symmetry)])
    y = np.array([label for label, _ in intensity_data])

    # Preprocess test set
    test_intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in test_set]
    test_symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in test_set]
    test_norm_intensity = data_processor.normalize_features(test_intensity_data)
    test_norm_symmetry = data_processor.normalize_features(test_symmetry_data)
    X_test = np.array([[intensity, symmetry] for (_, intensity), (_, symmetry) in zip(test_norm_intensity, test_norm_symmetry)])
    y_test = np.array([label for label, _ in test_intensity_data])

    # Binary classification (filtering only two classes for simplicity)
    class_1 = 1
    class_2 = -1
    train_indices = (y == class_1) | (y == class_2)
    test_indices = (y_test == class_1) | (y_test == class_2)
    X = X[train_indices]
    y = y[train_indices]
    y = np.where(y == class_1, 1, -1)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    y_test = np.where(y_test == class_1, 1, -1)

    # Small value of C
    model_small_C = ManualSVM(C=0.1, kernel=polynomial_kernel)
    model_small_C.fit(X, y)
    plot_decision_boundary(X, y, model_small_C, "Decision Boundary (C=0.1, Polynomial Kernel)")
    predictions_small_C = model_small_C.predict(X_test)
    E_test_small_C = np.mean(predictions_small_C != y_test)
    print(f"E_test (C=0.1): {E_test_small_C:.4f}")

    # Large value of C
    model_large_C = ManualSVM(C=100, kernel=polynomial_kernel)
    model_large_C.fit(X, y)
    plot_decision_boundary(X, y, model_large_C, "Decision Boundary (C=100, Polynomial Kernel)")
    predictions_large_C = model_large_C.predict(X_test)
    E_test_large_C = np.mean(predictions_large_C != y_test)
    print(f"E_test (C=100): {E_test_large_C:.4f}")

# Part (c): Perform cross-validation to find optimal C
def part_c(data_processor, train_set, test_set):
    # Randomly sample 300 points for training
    sampled_data = random.sample(train_set, 300)

    # Extract features (intensity and symmetry)
    intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in sampled_data]
    symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in sampled_data]
    norm_intensity = data_processor.normalize_features(intensity_data)
    norm_symmetry = data_processor.normalize_features(symmetry_data)

    # Combine features for SVM
    X = np.array([[intensity, symmetry] for (_, intensity), (_, symmetry) in zip(norm_intensity, norm_symmetry)])
    y = np.array([label for label, _ in intensity_data])

    # Preprocess test set
    test_intensity_data = [(label, data_processor.calculate_intensity(img)) for label, img in test_set]
    test_symmetry_data = [(label, data_processor.calculate_symmetry(img)) for label, img in test_set]
    test_norm_intensity = data_processor.normalize_features(test_intensity_data)
    test_norm_symmetry = data_processor.normalize_features(test_symmetry_data)
    X_test = np.array([[intensity, symmetry] for (_, intensity), (_, symmetry) in zip(test_norm_intensity, test_norm_symmetry)])
    y_test = np.array([label for label, _ in test_intensity_data])

    # Binary classification (filtering only two classes for simplicity)
    class_1 = 1
    class_2 = -1
    train_indices = (y == class_1) | (y == class_2)
    test_indices = (y_test == class_1) | (y_test == class_2)
    X = X[train_indices]
    y = y[train_indices]
    y = np.where(y == class_1, 1, -1)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    y_test = np.where(y_test == class_1, 1, -1)

    # Perform cross-validation to find the optimal C
    C_values = [0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    optimal_C = find_optimal_C(X, y, C_values, kernel=polynomial_kernel)

    # Train the final model with the optimal C
    model_optimal = ManualSVM(C=optimal_C, kernel=polynomial_kernel)
    model_optimal.fit(X, y)

    # Plot decision boundary for the optimal C
    plot_decision_boundary(X, y, model_optimal, f"Decision Boundary (Optimal C={optimal_C})")

    # Compute E_test for the optimal model
    predictions_test = model_optimal.predict(X_test)
    E_test_optimal = np.mean(predictions_test != y_test)
    print(f"E_test (Optimal C={optimal_C}): {E_test_optimal:.4f}")

# Main function
def main():
    data_processor = DataProcessor()
    train_set = data_processor.load_data('./ZipDigits.train')
    test_set = data_processor.load_data('./ZipDigits.test')

    # Run part (a)
    #print("Running part (a):")
    #part_a(data_processor, train_set, test_set)

    # Run part (c)
    print("\nRunning part (c):")
    part_c(data_processor, train_set, test_set)

if __name__ == "__main__":
    main()
