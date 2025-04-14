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
        labels.append(1 if label == 1 else 0)  # 1 for digit 1, 0 for digit 5 (Logistic regression works with 0/1 labels)
    return feature_data, labels

# Prepare labeled data for training and testing
train_features, train_labels = create_labeled_data(train_data, calculate_intensity, calculate_symmetry)
test_features, test_labels = create_labeled_data(test_data, calculate_intensity, calculate_symmetry)

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

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression with stochastic gradient descent (SGD)
def logistic_regression_sgd(features, labels, learning_rate=0.01, max_epochs=100):
    # Add bias term (augment features)
    augmented_features = np.hstack([np.ones((len(features), 1)), np.array(features)])
    weights = np.zeros(augmented_features.shape[1])  # Initialize weights

    for epoch in range(max_epochs):
        # Shuffle the data
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        augmented_features = augmented_features[indices]
        labels = np.array(labels)[indices]

        # Iterate through each training example
        for i in range(len(labels)):
            xi = augmented_features[i]  # Single example
            yi = labels[i]  # Corresponding label
            
            # Prediction using the sigmoid function
            prediction = sigmoid(np.dot(xi, weights))
            
            # Gradient for this example
            gradient = (prediction - yi) * xi
            
            # Update weights using SGD
            weights -= learning_rate * gradient

    return weights

# Calculate accuracy of the model
def calculate_accuracy(weights, data):
    feature_data, labels = data
    augmented_features = np.hstack([np.ones((len(feature_data), 1)), np.array(feature_data)])
    predictions = sigmoid(np.dot(augmented_features, weights)) >= 0.5
    correct_predictions = np.sum(predictions == labels)
    return correct_predictions / len(feature_data)

# Logistic regression model training and decision boundary plotting
def train_and_plot_logistic_regression(train_features, train_labels, test_features, test_labels):
    # Train logistic regression model using SGD
    weights = logistic_regression_sgd(train_features, train_labels)

    # Prepare the ranges for the contour plot
    intensity_range = np.linspace(-1, 0.3, 100)  # Intensity values
    symmetry_range = np.linspace(0, 0.6, 100)  # Symmetry values
    X1, X2 = np.meshgrid(intensity_range, symmetry_range)
    
    # Calculate the decision boundary
    def decision_boundary(x1, x2, weights):
        return weights[0] + weights[1] * x1 + weights[2] * x2

    # Draw the decision boundary on top of the training and testing data
    def draw_seps(weights):
        # Plot features for the training data
        fig1, ax1 = plot_features(train_data)
        cs1 = ax1.contour(X1, X2, decision_boundary(X1, X2, weights), levels=[0], colors='green')
        ax1.clabel(cs1, inline=True, fontsize=8, fmt="Logistic")
        ax1.set_title("Training Data")
        
        # Plot features for the testing data
        fig2, ax2 = plot_features(test_data)
        cs2 = ax2.contour(X1, X2, decision_boundary(X1, X2, weights), levels=[0], colors='green')
        ax2.clabel(cs2, inline=True, fontsize=8, fmt="Logistic")
        ax2.set_title("Testing Data with Training Result")
        
        # Plot decision boundary (hypothesis line) in green
        x_vals = np.linspace(-1, 0.3, 100)
        y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
        ax1.plot(x_vals, y_vals, 'green', label="Hypothesis")
        ax2.plot(x_vals, y_vals, 'green', label="Hypothesis")
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')

        plt.show()

    # Draw the decision boundary
    draw_seps(weights)

    # Calculate and print accuracy
    train_accuracy = calculate_accuracy(weights, (train_features, train_labels))
    test_accuracy = calculate_accuracy(weights, (test_features, test_labels))
    print(f"E_in: {1 - train_accuracy}, E_test: {1 - test_accuracy}")

# Train the model and plot the decision boundary using SGD
train_and_plot_logistic_regression(train_features, train_labels, test_features, test_labels)

# PART E: Third-order feature transformation and logistic regression using SGD

# Apply third-order feature transformation
def third_order_transform(features, labels):
    transformed_features = [
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
    ]
    return transformed_features, labels

# Transform the training and testing data using third-order transformation
transformed_train_features, transformed_train_labels = third_order_transform(train_features, train_labels)
transformed_test_features, transformed_test_labels = third_order_transform(test_features, test_labels)

# Train the logistic regression model using SGD on third-order transformed data
transformed_weights = logistic_regression_sgd(transformed_train_features, transformed_train_labels)

# Decision boundary for third-order transformation (hypothesis function)
def third_order_decision_boundary(x1, x2, weights):
    return (
        weights[0] + weights[1] * x1 + weights[2] * x2 +
        weights[3] * x1**2 + weights[4] * x2**2 +
        weights[5] * x1 * x2 + weights[6] * x1**3 +
        weights[7] * x2**3 + weights[8] * x1**2 * x2 +
        weights[9] * x1 * x2**2
    )

# Draw decision boundary for third-order transformed data
def draw_third_order_seps(weights):
    # Prepare the meshgrid for plotting
    intensity_range = np.linspace(-1, 0.5, 100)
    symmetry_range = np.linspace(0.4, 1.0, 100)  # Limiting the symmetry range to remove bottom boundary
    X1, X2 = np.meshgrid(intensity_range, symmetry_range)
    
    # Plot for the training data
    fig1, ax1 = plot_features(train_data)
    cs1 = ax1.contour(X1, X2, third_order_decision_boundary(X1, X2, weights), levels=[0], colors='green')
    ax1.clabel(cs1, inline=True, fontsize=8, fmt="Logistic")
    ax1.set_title("Third-Order Training Data")
    
    # Plot for the testing data
    fig2, ax2 = plot_features(test_data)
    cs2 = ax2.contour(X1, X2, third_order_decision_boundary(X1, X2, weights), levels=[0], colors='green')
    ax2.clabel(cs2, inline=True, fontsize=8, fmt="Logistic")
    ax2.set_title("Third-Order Testing Data with Training Result")
    
    # Add decision boundary to the legend
    ax1.plot([], [], 'green', label="Third-Order Boundary")
    ax2.plot([], [], 'green', label="Third-Order Boundary")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Show the plots
    plt.show()

# Execute the drawing function to visualize the third-order decision boundary
draw_third_order_seps(transformed_weights)

# Print the error rates for the third-order transformation
train_accuracy = calculate_accuracy(transformed_weights, (transformed_train_features, transformed_train_labels))
test_accuracy = calculate_accuracy(transformed_weights, (transformed_test_features, transformed_test_labels))
print("For 3rd-order transformation (SGD), E_in: {}, E_test: {}".format(1 - train_accuracy, 1 - test_accuracy))
