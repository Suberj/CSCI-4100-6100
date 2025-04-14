import matplotlib.pyplot as plt
import numpy as np

# Load and filter the dataset for digits 1 and 5
def load_digit_data(file_path):
    digit_data = []
    with open(file_path) as file:
        for line in file:
            values = line.split()
            if len(values) < 256:  # Ignore bad/incomplete data
                continue
            digit = int(float(values[0]))  # The first value is the label
            if digit in (1, 5):  # Only keep digits 1 and 5
                pixel_values = [float(pixel) for pixel in values[1:]]
                reshaped_image = np.reshape(pixel_values, (-1, 16))
                digit_data.append((digit, reshaped_image))
    return digit_data

# Load training and test datasets
train_data = load_digit_data('ZipDigits.train')
test_data = load_digit_data('ZipDigits.test')

# Feature calculation: average pixel intensity
def calculate_intensity(image):
    return np.mean(image)

# Feature calculation: vertical symmetry
def calculate_symmetry(image):
    image = image.reshape(16, 16)
    left_half = image[:, :8]
    right_half = np.flip(image[:, 8:], axis=1)
    symmetry_score = 1 - np.sum(np.abs(left_half - right_half)) / np.sum(np.abs(image))
    return symmetry_score

# Label the data with the two chosen features and corresponding labels
def create_labeled_data(data, feature_func1, feature_func2):
    feature_data = []
    labels = []
    for label, image in data:
        features = [feature_func1(image), feature_func2(image)]
        feature_data.append(features)
        labels.append(1 if label == 1 else -1)  # 1 for digit 1, -1 for digit 5
    return feature_data, labels

# Prepare labeled data for training and testing
train_features, train_labels = create_labeled_data(train_data, calculate_intensity, calculate_symmetry)
test_features, test_labels = create_labeled_data(test_data, calculate_intensity, calculate_symmetry)

# Linear regression function to calculate weights
def perform_linear_regression(features, labels):
    augmented_features = [[1] + feature for feature in features]  # Add bias term
    X = np.array(augmented_features)
    Y = np.array(labels)
    
    X_transposed = np.transpose(X)
    XTX = np.dot(X_transposed, X)
    XTX_inverse = np.linalg.inv(XTX)
    weights = np.dot(np.dot(XTX_inverse, X_transposed), Y)
    
    return weights

# Run Perceptron Learning Algorithm (PLA) once
def apply_perceptron_learning(features, labels, initial_weights):
    updated_weights = np.array(initial_weights)
    
    for feature, label in zip(features, labels):
        augmented_feature = np.array([1] + feature)
        if np.dot(updated_weights, augmented_feature) * label <= 0:
            updated_weights += label * augmented_feature  # Adjust weights on misclassification
            break  # Stop after first misclassification
    
    return updated_weights

# Calculate accuracy of the model
def calculate_accuracy(weights, data):
    feature_data, labels = data
    correct_predictions = 0
    
    for features, label in zip(feature_data, labels):
        prediction = np.dot(weights, [1] + features)  # [1] is the bias term
        if prediction * label > 0:
            correct_predictions += 1
    
    return correct_predictions / len(feature_data)

# Pocket algorithm to improve weights using PLA
def pocket_algorithm(data, initial_weights, max_iterations=10000):
    best_weights = initial_weights
    best_accuracy = calculate_accuracy(best_weights, data)
    
    feature_data, labels = data
    for _ in range(max_iterations):
        current_weights = apply_perceptron_learning(feature_data, labels, best_weights)
        current_accuracy = calculate_accuracy(current_weights, data)
        if current_accuracy > best_accuracy:
            best_weights = current_weights
            best_accuracy = current_accuracy
    
    return best_weights

# Plot intensity vs symmetry with color-coded labels
def plot_features(data):
    fig, ax = plt.subplots()
    intensity_1, symmetry_1 = [], []
    intensity_5, symmetry_5 = [], []
    
    for label, image in data:
        if label == 1:
            intensity_1.append(calculate_intensity(image))
            symmetry_1.append(calculate_symmetry(image))
        else:
            intensity_5.append(calculate_intensity(image))
            symmetry_5.append(calculate_symmetry(image))
    
    ax.plot(intensity_1, symmetry_1, 'bo', label="Digit 1")  # Blue circles for digit 1
    ax.plot(intensity_5, symmetry_5, 'rx', label="Digit 5")  # Red crosses for digit 5
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Symmetry")
    
    return fig, ax

# Calculate the decision boundary (hypothesis)
def decision_boundary(x1, x2, weights):
    return weights[0] + weights[1] * x1 + weights[2] * x2

# Prepare the ranges for the contour plot
intensity_range = np.linspace(-1, 0.3, 100)  # Intensity values
symmetry_range = np.linspace(0, 0.6, 100)  # Symmetry values
X1, X2 = np.meshgrid(intensity_range, symmetry_range)
levels = [0]

# Draw the decision boundary on top of the training and testing data
def draw_seps(weights):
    # Plot features for the training data
    fig1, ax1 = plot_features(train_data)
    cs1 = ax1.contour(X1, X2, decision_boundary(X1, X2, weights), levels=levels)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt="LR + Pocket")
    ax1.set_title("Training Data")
    
    # Plot features for the testing data
    fig2, ax2 = plot_features(test_data)
    cs2 = ax2.contour(X1, X2, decision_boundary(X1, X2, weights), levels=levels)
    ax2.clabel(cs2, inline=True, fontsize=8, fmt="LR + Pocket")
    ax2.set_title("Testing Data with Training result")

    # Plot decision boundary (hypothesis line) in green
    x_vals = np.linspace(-1, 0.3, 100)
    y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
    ax1.plot(x_vals, y_vals, 'green', label="Hypothesis")
    ax2.plot(x_vals, y_vals, 'green', label="Hypothesis")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.show()

# Calculate linear regression weights and improve them using pocket algorithm
def train_model(train_data):
    initial_weights = perform_linear_regression(train_data[0], train_data[1])
    optimal_weights = pocket_algorithm(train_data, initial_weights)
    return optimal_weights

# Train the model and draw the decision boundary
trained_weights = train_model((train_features, train_labels))
draw_seps(trained_weights)

# Calculate and print the training and testing errors
train_accuracy = calculate_accuracy(trained_weights, (train_features, train_labels))
test_accuracy = calculate_accuracy(trained_weights, (test_features, test_labels))
print(f"E_in: {1 - train_accuracy}, E_test: {1 - test_accuracy}")

# Print the size of the training and test datasets
print(f"Training set size: {len(train_features)}, Test set size: {len(test_features)}")

#######################################################################################

# PART E

# Apply third-order feature transformation
def third_order_transform(features, labels):
    return [
        [
            x1, 
            x2, 
            x1**2, 
            x2**2, 
            x1 * x2, 
            x1**3, 
            x2**3, 
            x1**2 * x2, 
            x1 * x2**2
        ] for x1, x2 in features
    ], labels

# Transform the training and testing data
transformed_train_data = third_order_transform(train_features, train_labels)
transformed_test_data = third_order_transform(test_features, test_labels)

# Train the model using the transformed training data
transformed_weights = pocket_algorithm(transformed_train_data, perform_linear_regression(transformed_train_data[0], transformed_train_data[1]))

# Calculate the third-order decision boundary (hypothesis function)
def third_order_decision_boundary(x1, x2, weights):
    return (
        weights[0] + weights[1] * x1 + weights[2] * x2 +
        weights[3] * x1**2 + weights[4] * x2**2 +
        weights[5] * x1 * x2 + weights[6] * x1**3 +
        weights[7] * x2**3 + weights[8] * x1**2 * x2 +
        weights[9] * x1 * x2**2
    )

# Prepare the meshgrid for plotting
intensity_range = np.linspace(-1, 0.5, 100)
symmetry_range = np.linspace(0.4, 1.0, 100)  # Limiting the symmetry range to avoid the bottom line
X1, X2 = np.meshgrid(intensity_range, symmetry_range)

# Draw the third-order decision boundary for training and testing data
def plot_decision_boundary(weights):
    # Plot for the training data
    fig1, ax1 = plot_features(train_data)
    cs1 = ax1.contour(X1, X2, third_order_decision_boundary(X1, X2, weights), levels=[0], colors='green')
    ax1.clabel(cs1, inline=True, fontsize=8, fmt="Sep")
    ax1.set_title("Third-Order Training Data")
    
    # Plot for the testing data
    fig2, ax2 = plot_features(test_data)
    cs2 = ax2.contour(X1, X2, third_order_decision_boundary(X1, X2, weights), levels=[0], colors='green')
    ax2.clabel(cs2, inline=True, fontsize=8, fmt="Sep")
    ax2.set_title("Third-Order Testing Data with Training Result")
    
    # Add decision boundary to the legend
    ax1.plot([], [], 'green', label="Sep")
    ax2.plot([], [], 'green', label="Sep")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Display the plots
    plt.show()

# Execute the function to visualize the decision boundary
plot_decision_boundary(transformed_weights)

# Print the error rates for the third-order transformation
print("For 3rd-order transformation, E_in: {}, E_test: {}".format(
    1 - calculate_accuracy(transformed_weights, transformed_train_data),
    1 - calculate_accuracy(transformed_weights, transformed_test_data)
))