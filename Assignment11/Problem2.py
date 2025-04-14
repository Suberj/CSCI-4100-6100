import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data Extraction with Digit Labeling
def load_digit_data(filename):
    digit_data = []
    with open(filename) as file:
        for line in file:
            columns = line.split()
            if len(columns) < 256:
                continue
            digit_label = int(float(columns[0]))
            pixel_values = np.array([float(pixel) for pixel in columns[1:]])
            digit_data.append((1 if digit_label == 1 else -1, pixel_values))
    return digit_data

# Feature Extraction
def calculate_intensity(image):
    return np.mean(image)

def calculate_symmetry(image):
    image = image.reshape(16, 16)
    left_half = image[:, :8]
    right_half = np.flip(image[:, 8:], axis=1)
    symmetry_score = 1 - np.sum(np.abs(left_half - right_half)) / np.sum(np.abs(image))
    return symmetry_score

# Normalize features
def normalize_features(features, target_max=1.0, target_min=-1.0):
    feature_values = [value for (_, value) in features]
    original_max, original_min = max(feature_values), min(feature_values)
    scale_factor = (target_max - target_min) / (original_max - original_min)
    shift_offset = (target_max + target_min) / 2 - (original_max + original_min) / 2
    return [(label, (value + shift_offset) * scale_factor) for (label, value) in features]

# Split dataset randomly
def split_random_data(*feature_sets, train_size=300):
    data_combined = [(label, tuple(features[i][1] for features in feature_sets)) for i, (label, _) in enumerate(feature_sets[0])]
    random.shuffle(data_combined)
    return data_combined[:train_size], data_combined[train_size:]

# RBF Network with Gaussian Kernel
class RBFNetwork:
    def __init__(self, num_centers, r):
        self.num_centers = num_centers  # Number of RBF centers
        self.r = r                      # Spread of the Gaussian kernel
        self.centers = None             # RBF centers
        self.weights = None             # Linear weights

    def _gaussian_kernel(self, x, center):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.r ** 2))

    def _compute_design_matrix(self, X):
        # Compute the design matrix (Phi)
        phi = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                phi[i, j] = self._gaussian_kernel(x, center)
        return phi

    def fit(self, X, y):
        # Select RBF centers using KMeans
        kmeans = KMeans(n_clusters=self.num_centers, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_

        # Compute design matrix
        phi = self._compute_design_matrix(X)

        # Solve for weights using least squares
        self.weights = np.linalg.pinv(phi.T @ phi) @ phi.T @ y

    def predict(self, X):
        # Compute design matrix
        phi = self._compute_design_matrix(X)
        return np.sign(phi @ self.weights)

# Cross-validation for RBF Network
def cross_validation_rbf(training_set, k_values, num_folds=5):
    random.shuffle(training_set)
    fold_size = len(training_set) // num_folds
    folds = [training_set[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    cv_errors = []

    for k in k_values:
        fold_errors = []
        r = 2 / np.sqrt(k)  # Set scale parameter
        for i in range(num_folds):
            validation_set = folds[i]
            train_set = [item for s in folds if s is not validation_set for item in s]

            train_features = np.array([features for (_, features) in train_set])
            train_labels = np.array([label for (label, _) in train_set])

            val_features = np.array([features for (_, features) in validation_set])
            val_labels = np.array([label for (label, _) in validation_set])

            # Train RBF Network
            rbf = RBFNetwork(num_centers=k, r=r)
            rbf.fit(train_features, train_labels)

            # Predict on validation set
            predictions = rbf.predict(val_features)
            fold_error = np.mean(predictions != val_labels)
            fold_errors.append(fold_error)

        cv_errors.append(np.mean(fold_errors))
    return cv_errors

# Plot decision boundary for RBF Network
def plot_rbf_decision_boundary(training_set, rbf, feature_range=(-1, 1), resolution=100):
    x_min, x_max = feature_range
    y_min, y_max = feature_range
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    predictions = rbf.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.7, cmap='coolwarm')

    # Plot training points
    training_labels = np.array([label for (label, _) in training_set])
    training_features = np.array([features for (_, features) in training_set])
    plt.scatter(
        training_features[training_labels == 1][:, 0],
        training_features[training_labels == 1][:, 1],
        c='blue', label='Digit 1', edgecolor='k'
    )
    plt.scatter(
        training_features[training_labels == -1][:, 0],
        training_features[training_labels == -1][:, 1],
        c='red', label='Not Digit 1', edgecolor='k'
    )

    plt.title(f"Decision Boundary for RBF Network (k={rbf.num_centers})")
    plt.xlabel("Feature 1 (Normalized Intensity)")
    plt.ylabel("Feature 2 (Normalized Symmetry)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load and preprocess the data (reusing the functions from k-NN)
    train_data = load_digit_data('ZipDigits.train')
    test_data = load_digit_data('ZipDigits.test')
    all_data = train_data + test_data

    # Feature extraction and normalization
    intensity_features = [(label, calculate_intensity(image)) for (label, image) in all_data]
    symmetry_features = [(label, calculate_symmetry(image)) for (label, image) in all_data]
    normalized_intensity = normalize_features(intensity_features)
    normalized_symmetry = normalize_features(symmetry_features)

    # Split into training and testing sets
    training_set, testing_set = split_random_data(normalized_intensity, normalized_symmetry, train_size=250)

    # Define range of k values
    k_values = range(2, 21)

    # Perform cross-validation
    cv_errors = cross_validation_rbf(training_set, k_values)

    # Determine optimal k
    optimal_k = k_values[np.argmin(cv_errors)]
    print(f"Optimal k: {optimal_k}")

    # Plot E_cv versus k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, cv_errors, marker='o', color='blue', linestyle='-', linewidth=0.8, label="E_cv")
    plt.title("Cross Validation Error of RBF Network ($E_{cv}$ vs. k)")
    plt.xlabel("k (Number of Centers)")
    plt.ylabel("Cross-Validation Error ($E_{cv}$)")
    plt.grid(True)

    # Highlight the optimal k
    plt.axvline(optimal_k, color='red', linestyle='--', label=f"Optimal k: {optimal_k}")
    plt.legend()
    plt.show()

    # Train and plot decision boundary for optimal k
    train_features = np.array([features for (_, features) in training_set])
    train_labels = np.array([label for (label, _) in training_set])
    r = 2 / np.sqrt(optimal_k)
    rbf = RBFNetwork(num_centers=optimal_k, r=r)
    rbf.fit(train_features, train_labels)
    plot_rbf_decision_boundary(training_set, rbf)

    # Calculate in-sample error
    in_sample_predictions = rbf.predict(train_features)
    in_sample_error = np.mean(in_sample_predictions != train_labels)
    print(f"In-Sample Error: {in_sample_error}")

    # Calculate test error
    test_features = np.array([features for (_, features) in testing_set])
    test_labels = np.array([label for (label, _) in testing_set])
    test_predictions = rbf.predict(test_features)
    test_error = np.mean(test_predictions != test_labels)
    print(f"Test Error (E_test): {test_error}")
