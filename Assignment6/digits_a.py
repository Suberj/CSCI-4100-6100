import numpy as np
import matplotlib.pyplot as plt

# Load the data while skipping bad lines
def load_digits_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.split()
            # Check if row has the correct number of columns (1 label + 256 pixel values)
            if len(row) == 257:
                data.append([float(x) for x in row])
    return np.array(data)

# Load the cleaned data
train_file = 'ZipDigits_train_1_5_only.txt'
data = load_digits_file(train_file)

# Separate digits and pixel values
digits = data[:, 0]
images = data[:, 1:]

# Function to plot the digit image
def plot_digit(image, label, ax):
    ax.imshow(image.reshape(16, 16), cmap='gray', interpolation='nearest')
    ax.set_title(f'Digit: {int(label)}')
    ax.axis('off')

# Select two example images (one for digit 1, one for digit 5)
digit_1 = images[digits == 1][0]
digit_5 = images[digits == 5][0]

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
plot_digit(digit_1, 1, axs[0])
plot_digit(digit_5, 5, axs[1])

# Show the plot
plt.tight_layout()
plt.show()
