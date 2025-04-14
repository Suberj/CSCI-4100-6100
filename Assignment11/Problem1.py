
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

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

# Compute distances
def compute_distances(train_set, test_point):
    return [np.linalg.norm(np.array(features) - np.array(test_point)) for (_, features) in train_set]

# k-NN Prediction
def knn_predict(train_set, test_point, k):
    distances = compute_distances(train_set, test_point)
    nearest_neighbors = sorted(zip(distances, train_set))[:k]
    labels = [label for (_, (label, _)) in nearest_neighbors]
    return Counter(labels).most_common(1)[0][0]

# Cross-validation
def cross_validation(training_set, k_values, num_folds=5):
    random.shuffle(training_set)
    fold_size = len(training_set) // num_folds
    folds = [training_set[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    cv_errors = []

    for k in k_values:
        fold_errors = []
        for i in range(num_folds):
            validation_set = folds[i]
            training_set = [item for s in folds if s is not validation_set for item in s]
            errors = 0
            for label, features in validation_set:
                prediction = knn_predict(training_set, features, k)
                if prediction != label:
                    errors += 1
            fold_errors.append(errors / len(validation_set))
        cv_errors.append(np.mean(fold_errors))
    return cv_errors

# Plot decision boundary
def plot_decision_boundary(training_set, k, feature_range=(-1, 1), resolution=100):
    # Generate a grid of points over the feature space
    x_min, x_max = feature_range
    y_min, y_max = feature_range
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Classify each grid point using k-NN
    grid_predictions = []
    for point in grid_points:
        prediction = knn_predict(training_set, point, k)
        grid_predictions.append(prediction)

    # Reshape predictions to match the grid
    grid_predictions = np.array(grid_predictions).reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_predictions, alpha=0.7, cmap='coolwarm')

    # Overlay the training points
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

    plt.title(f"Decision Boundary for k-NN (k={k})")
    plt.xlabel("Feature 1 (Normalized Intensity)")
    plt.ylabel("Feature 2 (Normalized Symmetry)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Calculate in-sample error
def calculate_in_sample_error(training_set, k):
    errors = 0
    for label, features in training_set:
        prediction = knn_predict(training_set, features, k)
        if prediction != label:
            errors += 1
    return errors / len(training_set)

# Calculate test error
def calculate_test_error(test_set, training_set, k):
    errors = 0
    for label, features in test_set:
        prediction = knn_predict(training_set, features, k)
        if prediction != label:
            errors += 1
    return errors / len(test_set)

# Main execution
if __name__ == "__main__":
    
    # Load and preprocess the data
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
    k_values = range(1, 51)

    # Perform cross-validation
    cv_errors = cross_validation(training_set, k_values)

    # Determine optimal k
    optimal_k = k_values[np.argmin(cv_errors)]
    print(f"Optimal k: {optimal_k}")

    # Plot E_cv versus k
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, cv_errors, marker='o', color='blue', linestyle='-', linewidth=0.8, label="E_cv")
    plt.title("Cross Validation Error of k-NN ($E_{cv}$ vs. k)")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Cross-Validation Error ($E_{cv}$)")
    plt.grid(True)

    # Highlight the optimal k
    plt.axvline(optimal_k, color='red', linestyle='--', label=f"Optimal k: {optimal_k}")
    plt.legend()
    plt.show()

    # Plot decision boundary for the optimal k
    plot_decision_boundary(training_set, k=optimal_k)

    # Calculate and print in-sample error
    in_sample_error = calculate_in_sample_error(training_set, optimal_k)
    print(f"In-Sample Error: {in_sample_error}")

    # Calculate and print test error
    test_error = calculate_test_error(testing_set, training_set, optimal_k)
    print(f"Test Error (E_test): {test_error}")
