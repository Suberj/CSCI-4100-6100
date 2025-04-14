import random
import numpy as np
import matplotlib.pyplot as plt

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

# Load and combine training and testing data
train_data = load_digit_data('ZipDigits.train')
test_data = load_digit_data('ZipDigits.test')
all_data = train_data + test_data

# Feature Extraction: Updated Symmetry and Intensity Calculations
def calculate_intensity(image):
    return np.mean(image)

def calculate_symmetry(image):
    image = image.reshape(16, 16)
    left_half = image[:, :8]
    right_half = np.flip(image[:, 8:], axis=1)
    symmetry_score = 1 - np.sum(np.abs(left_half - right_half)) / np.sum(np.abs(image))
    return symmetry_score

# Extract features and normalize
intensity_features = [(label, calculate_intensity(image)) for (label, image) in all_data]
symmetry_features = [(label, calculate_symmetry(image)) for (label, image) in all_data]

def normalize_features(features, target_max=1.0, target_min=-1.0):
    feature_values = [value for (_, value) in features]
    original_max, original_min = max(feature_values), min(feature_values)
    scale_factor = (target_max - target_min) / (original_max - original_min)
    shift_offset = (target_max + target_min) / 2 - (original_max + original_min) / 2
    return [(label, (value + shift_offset) * scale_factor) for (label, value) in features]

normalized_intensity = normalize_features(intensity_features)
normalized_symmetry = normalize_features(symmetry_features)

# Randomly split data into training and testing sets
def split_random_data(*feature_sets, train_size=300):
    data_combined = [(label, tuple(features[i][1] for features in feature_sets)) for i, (label, _) in enumerate(feature_sets[0])]
    random.shuffle(data_combined)
    return data_combined[:train_size], data_combined[train_size:]

training_set, testing_set = split_random_data(normalized_intensity, normalized_symmetry)

# Regularized Linear Regression for Classification
def compute_regularized_weights(dataset, regularization_param):
    features = [features for (_, features) in dataset]
    labels = np.array([label for (label, _) in dataset]).T
    dimensions = len(features[0])
    
    identity_matrix = np.identity(dimensions)
    feature_transpose = np.transpose(features)
    reg_term = regularization_param * identity_matrix
    weight_matrix = np.linalg.inv(np.dot(feature_transpose, features) + reg_term).dot(feature_transpose).dot(labels)
    
    return weight_matrix

# Feature Visualization with Labels
def plot_data_features(*datasets):
    fig, ax = plt.subplots()
    positive_points, negative_points = [], []
    for dataset in datasets:
        for label, feature_pair in dataset:
            if label == 1:
                positive_points.append(feature_pair)
            else:
                negative_points.append(feature_pair)
    
    ax.scatter(*zip(*positive_points), marker="o", label="Digit 1")
    ax.scatter(*zip(*negative_points), marker="x", label="Other Digits")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Symmetry")
    ax.legend(loc='upper right')  # Place legend within the plot at the top-right corner
    return fig, ax

# Generate Legendre Polynomials for Feature Transformation
def legendre_polynomial(order, x_value):
    cache = [None] * (order + 1)
    return _legendre_polynomial_recursive(order, x_value, cache)

def _legendre_polynomial_recursive(order, x_value, cache):
    if order == 0:
        return 1
    elif order == 1:
        return x_value
    elif cache[order] is not None:
        return cache[order]
    else:
        value = ((2 * order - 1) * x_value * _legendre_polynomial_recursive(order - 1, x_value, cache) - (order - 1) * _legendre_polynomial_recursive(order - 2, x_value, cache)) / order
        cache[order] = value
        return value

# Generate Coefficients for Legendre Polynomial
coefficients_cache = [None] * 100
def generate_coefficients(order):
    if coefficients_cache[order] is not None:
        return coefficients_cache[order]
    coefficients = [(order - i, i) for i in range(order + 1)]
    coefficients_cache[order] = coefficients
    return coefficients

def transform_with_legendre(order, x1, x2):
    coeff_pairs = sum([generate_coefficients(i) for i in range(order + 1)], [])
    return [legendre_polynomial(i, x1) * legendre_polynomial(j, x2) for (i, j) in coeff_pairs]

# Transform dataset for polynomial-based feature expansion
polynomial_order = 8
train_transformed = [(label, transform_with_legendre(polynomial_order, feature[0], feature[1])) for (label, feature) in training_set]
test_transformed = [(label, transform_with_legendre(polynomial_order, feature[0], feature[1])) for (label, feature) in testing_set]

def plot_decision_boundary(weights, title, lambda_value, *datasets):
    fig, ax = plot_data_features(*datasets)
    # Adjust range to better fit data bounds
    x1_range, x2_range = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)  # Adjusted to -1 to 1 range
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    # Transform features with the legendre polynomial
    transformed_features = [transform_with_legendre(polynomial_order, x1, x2) for (x1, x2) in zip(x1_grid.ravel(), x2_grid.ravel())]
    decision_values = np.array([sum(coef * feature for coef, feature in zip(weights, feature)) for feature in transformed_features])
    decision_values = decision_values.reshape(x1_grid.shape)
    
    # Plot contour within data bounds and adjust levels
    contour = ax.contour(x1_grid, x2_grid, decision_values, levels=[0], colors='purple')
    contour.collections[0].set_label(f"Lambda = {lambda_value}")
    ax.set_title(title)
    ax.legend(loc='upper right')  # Keep legend in the top-right
    plt.show()

#############
# Question 2
#############
# Train and Plot for Different Regularization
weights_overfit = compute_regularized_weights(train_transformed, 0)
plot_decision_boundary(weights_overfit, "Overfitting", 0, training_set)

#############
# Question 3
#############
weights_regularized = compute_regularized_weights(train_transformed, 2)
plot_decision_boundary(weights_regularized, "Regularization", 2, training_set)

#############
# Question 4
#############
# Cross-Validation Error Calculation
def compute_cross_validation_error(data, regularization_lambda):
    features = [z for (_, z) in data]
    labels = np.array([y for (y, _) in data]).T
    feature_dim = len(features[0])
    
    identity_matrix = np.identity(feature_dim)
    features_transpose = np.transpose(features)
    ZTZ = np.dot(features_transpose, features)
    regularization_matrix = regularization_lambda * identity_matrix
    ZTZ_reg_inv = np.linalg.inv(ZTZ + regularization_matrix)
    
    # Compute the hat matrix H and the predicted labels
    H_matrix = np.dot(np.dot(features, ZTZ_reg_inv), features_transpose)
    predictions = np.dot(H_matrix, labels)
    
    # Leave-One-Out Cross-Validation Error
    cv_error = 0.0
    total_samples = len(data)
    for i in range(total_samples):
        cv_error += ((float(predictions[i] - labels[i]) / (1 - H_matrix[i][i])) ** 2) / total_samples
    
    return cv_error

# Test Error Calculation based on Regression Error
def calculate_regression_error(weights, dataset):
    total_error = 0
    for label, features in dataset:
        prediction_error = (label - np.dot(weights, features)) ** 2
        total_error += prediction_error
    return float(total_error) / len(dataset)

# Run Cross-Validation over a range of regularization values
def perform_cross_validation():
    cv_errors = []
    test_errors = []
    lambda_values = np.arange(0, 2, 0.01)
    best_lambda = None
    min_cv_error = float('inf')

    for regularization_lambda in lambda_values:
        # Train model with regularization
        weights = compute_regularized_weights(train_transformed, regularization_lambda)
        
        # Calculate cross-validation and test errors
        cv_error = compute_cross_validation_error(train_transformed, regularization_lambda)
        test_error = calculate_regression_error(weights, test_transformed)
        
        # Track the lambda that gives the minimum cross-validation error
        if cv_error < min_cv_error:
            min_cv_error = cv_error
            best_lambda = regularization_lambda
        
        cv_errors.append(cv_error)
        test_errors.append(test_error)

    return cv_errors, test_errors, lambda_values, round(best_lambda, 2)

# Run cross-validation and obtain errors
cv_errors, test_errors, lambda_values, optimal_lambda = perform_cross_validation()

# Check the range of E_cv and E_test values
print("E_cv range:", min(cv_errors), "-", max(cv_errors))
print("E_test range:", min(test_errors), "-", max(test_errors))

# Plot Cross-Validation and Test Errors with a logarithmic y-axis
def plot_cross_validation_results_log(cv_errors, test_errors, lambda_values):
    fig, ax = plt.subplots()
    ax.scatter(lambda_values, cv_errors, color="red", marker="x", s=1, label="E_cv")
    ax.scatter(lambda_values, test_errors, color="blue", marker="o", s=1, label="E_test")
    ax.set_title("Cross-Validation and Test Error for Different Lambda Values")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Error")
    
    # Set a logarithmic scale on the y-axis to better visualize the spread
    ax.set_yscale('log')
    ax.legend()
    plt.show()

# Call the function to visualize the errors with logarithmic scale
plot_cross_validation_results_log(cv_errors, test_errors, lambda_values)

#############
# Question 5
#############
# Use optimal lambda from cross-validation
print("Optimal Lambda:", optimal_lambda)

# Train with the best lambda
best_weights = compute_regularized_weights(train_transformed, optimal_lambda)

# Plot decision boundaries with the chosen lambda for both training and testing sets
plot_decision_boundary(best_weights, "Decision Boundary with Optimal Lambda (Train Set)", optimal_lambda, training_set)
plt.show()  # Ensure the plot for the train set appears

#############
# Question 6
#############
# Calculate and display the test error with the chosen lambda
final_test_error = calculate_regression_error(best_weights, test_transformed)
num_test_samples = len(test_transformed)
print(f"Final Test Error (E_test): {final_test_error}, Number of Test Samples: {num_test_samples}")


