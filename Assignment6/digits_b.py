import numpy as np

# Load the data (assuming you've already cleaned and prepared the file with only 1s and 5s)
train_file = 'ZipDigits_train_1_5_only.txt'

# Load the data
def load_digits_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.split()
            if len(row) == 257:  # Ensure correct number of columns
                data.append([float(x) for x in row])
    return np.array(data)

# Load the dataset
data = load_digits_file(train_file)

# Separate digits and pixel values
digits = data[:, 0]
images = data[:, 1:]

# Define the two features: Average Intensity and Vertical Symmetry
def average_intensity(image):
    return np.mean(image)

def vertical_symmetry(image):
    image = image.reshape(16, 16)
    left_half = image[:, :8]
    right_half = image[:, 8:]
    right_half_flipped = np.flip(right_half, axis=1)
    symmetry_score = 1 - np.sum(np.abs(left_half - right_half_flipped)) / np.sum(np.abs(image))
    return symmetry_score

# Select two example images (one for digit 1, one for digit 5)
digit_1_image = images[digits == 1][0]
digit_5_image = images[digits == 5][0]

# Compute the features for each image
digit_1_intensity = average_intensity(digit_1_image)
digit_1_symmetry = vertical_symmetry(digit_1_image)

digit_5_intensity = average_intensity(digit_5_image)
digit_5_symmetry = vertical_symmetry(digit_5_image)

# Print the computed features
print(f"Digit 1 - Average Intensity: {digit_1_intensity}, Vertical Symmetry: {digit_1_symmetry}")
print(f"Digit 5 - Average Intensity: {digit_5_intensity}, Vertical Symmetry: {digit_5_symmetry}")
