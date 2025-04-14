import numpy as np

def perceptron_learning_algorithm(X, y):
    # Initialize weights to zeros (size depends on the number of features including the bias)
    weights = np.zeros(X.shape[1])
    
    iterations = 0  # Keep track of the number of updates
    converged = False  # Boolean flag for convergence

    while not converged:
        converged = True  # Assume we've converged
        for i in range(len(y)):
            # Prediction: sign of the dot product of weights and input X[i]
            prediction = np.sign(np.dot(weights, X[i]))

            # If the prediction is wrong, update the weights
            if prediction != y[i]:
                weights += y[i] * X[i]  # Perceptron update rule
                converged = False  # Set convergence to False since we updated
                iterations += 1  # Track the number of updates
                
    return weights, iterations
