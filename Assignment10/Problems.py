import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
import itertools
from matplotlib.colors import ListedColormap
from disk_utils import Coordinate, SeparatedDisk

################
# Question 6.1 #
################
# Define dataset with labeled points
sample_data = [([1, 0], -1), ([0, 1], -1), ([0, -1], -1), ([-1, 0], -1), 
               ([0, 2], 1), ([0, -2], 1), ([-2, 0], 1)]

# Utility functions
def compute_distance(point_a, point_b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

def get_sign(value):
    """Return the sign of a number as 1 or -1."""
    return 1 if value > 0 else -1

def polar_transform(coords):
    """Transform Cartesian coordinates to polar to avoid division by zero."""
    x, y = coords
    return [np.sqrt(x ** 2 + y ** 2), np.arctan(float(y) / (x + 1e-10))]

# KNN Classifier Function
def knn_classify(dataset, target_point, k_neighbors=1):
    """Perform k-nearest neighbor classification on a target point."""
    distances = [(compute_distance(target_point, coords), label) for coords, label in dataset]
    distances.sort(key=lambda pair: pair[0])
    nearest_labels = [label for _, label in distances[:k_neighbors]]
    return get_sign(sum(nearest_labels))

# KNN Grid Generation Function
def generate_knn_grid(dataset, x_range=(-3, 3), y_range=(-3, 3), k_neighbors=1, apply_transform=False):
    """Generate KNN grid for a range of points with optional coordinate transformation."""
    x_vals = np.linspace(x_range[0], x_range[1], 250)
    y_vals = np.linspace(y_range[0], y_range[1], 250)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    points_grid = [zip(x_row, y_row) for x_row, y_row in zip(x_grid, y_grid)]

    # Apply transformation if required
    if apply_transform:
        dataset = [(polar_transform(coords), label) for coords, label in dataset]
        points_grid = [[polar_transform(point) for point in row] for row in points_grid]

    # Classify each point in the grid
    classification_grid = [[knn_classify(dataset, point, k_neighbors) for point in row] for row in points_grid]
    
    return x_grid, y_grid, classification_grid

# Plotting functions with adjusted structure and new descriptions
def plot_data_points(data):
    """Visualize data points with unique markers based on their class labels."""
    fig, axis = plt.subplots()
    
    positive_points, negative_points = [], []

    # Separate points based on class
    for point, label in data:
        if label == 1:
            positive_points.append(point)
        else:
            negative_points.append(point)

    # Assign colors and markers (blue and orange updated)
    axis.scatter(*zip(*positive_points), color="blue", marker="o", label="Class +1", zorder=10)
    axis.scatter(*zip(*negative_points), color="orange", marker="x", label="Class -1", zorder=10)

    # Add axis labels
    axis.set_xlabel("X-axis")
    axis.set_ylabel("Y-axis")
    
    return fig, axis

# Visualization for Part A
def visualize_knn_no_transform():
    fig, axis = plot_data_points(sample_data)
    axis.contourf(X_no_trans_k1, Y_no_trans_k1, Z_no_trans_k1, cmap=cmap_regions, alpha=0.5)
    axis.set_title('1-Nearest Neighbor without Transformation')
    fig.legend()
    
    fig, axis = plot_data_points(sample_data)
    axis.contourf(X_no_trans_k3, Y_no_trans_k3, Z_no_trans_k3, cmap=cmap_regions, alpha=0.5)
    axis.set_title('3-Nearest Neighbor without Transformation')
    fig.legend()
    
    plt.show()

# Visualization for Part B
def visualize_knn_with_transform():
    fig, axis = plot_data_points(sample_data)
    axis.contourf(X_trans_k1, Y_trans_k1, Z_trans_k1, cmap=cmap_regions, alpha=0.5)
    axis.set_title('1-Nearest Neighbor with Transformation')
    fig.legend()
    
    fig, axis = plot_data_points(sample_data)
    axis.contourf(X_trans_k3, Y_trans_k3, Z_trans_k3, cmap=cmap_regions, alpha=0.5)
    axis.set_title('3-Nearest Neighbor with Transformation')
    fig.legend()
    
    plt.show()
    
# Generate classifications for different configurations
X_no_trans_k1, Y_no_trans_k1, Z_no_trans_k1 = generate_knn_grid(sample_data)
X_no_trans_k3, Y_no_trans_k3, Z_no_trans_k3 = generate_knn_grid(sample_data, k_neighbors=3)
X_trans_k1, Y_trans_k1, Z_trans_k1 = generate_knn_grid(sample_data, apply_transform=True)
X_trans_k3, Y_trans_k3, Z_trans_k3 = generate_knn_grid(sample_data, k_neighbors=3, apply_transform=True)

# Define custom colors for the regions
cmap_regions = ListedColormap(["#146eb4", "#ff9900"])  # Red and Cyan colors

##########################
# Question 6.4: Disk Fun #
##########################

# Experiment Function: Generates points and visualizes KNN regions
def execute_experiment(radius=10.0, thickness=5.0, separation=5.0, num_points=2000):
    """Run experiment to generate and plot points with KNN decision boundaries."""
    disk = SeparatedDisk(radius, thickness, separation)
    disk.generate_points(num_points)
    print(f"{len(disk.point_list)} Points generated")

    # Prepare data for KNN
    knn_data = [(point, label) for point, label in zip(disk.point_list, disk.labels)]
    region_colors = ListedColormap(["#ff9999", "#99ffff"])  # Red and Cyan

    # 1-NN Decision Boundary Plot
    X_1nn, Y_1nn, Z_1nn = generate_knn_grid(knn_data, x_range=(-5, 50), y_range=(-5, 40), k_neighbors=1)
    fig, ax = plt.subplots()
    ax.contourf(X_1nn, Y_1nn, Z_1nn, cmap=region_colors, alpha=0.5)
    ax.set_title('1-NN Decision Boundary')
    disk.plot_points(ax)
    fig.legend()

    # 3-NN Decision Boundary Plot
    X_3nn, Y_3nn, Z_3nn = generate_knn_grid(knn_data, x_range=(-5, 50), y_range=(-5, 40), k_neighbors=3)
    fig, ax = plt.subplots()
    ax.contourf(X_3nn, Y_3nn, Z_3nn, cmap=region_colors, alpha=0.5)
    ax.set_title('3-NN Decision Boundary')
    disk.plot_points(ax)
    fig.legend()

    plt.show()

################
# Question 6.16 #
################

# Function to generate uniformly distributed random data points
def generate_uniform_data(x_min, x_max, y_min, y_max, count):
    return [((np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)), 1) for _ in range(count)]
uniform_data = generate_uniform_data(0, 1, 0, 1, 10000)
uniform_points = [point for point, _ in uniform_data]

# Simple Center Selection
def select_simple_centers(data, num_centers):
    """Select initial centers using a simple greedy heuristic."""
    first_center = data[np.random.randint(0, len(data))][0]
    centers = [first_center]
    
    for _ in range(num_centers - 1):
        farthest_point = None
        max_distance = 0
        for point, _ in data:
            # Find the minimum distance to the existing centers
            min_distance_to_centers = min(compute_distance(point, center) for center in centers)
            if min_distance_to_centers > max_distance:
                farthest_point = point
                max_distance = min_distance_to_centers
        centers.append(farthest_point)
    return centers

# Utility functions for clustering
def average(values):
    return sum(values) / len(values)
def split_coordinates(points):
    return [x for x, _ in points], [y for _, y in points]
def index_of_min(values):
    return values.index(min(values))

# Clustering with Lloyd's Update Algorithm
def lloyd_clustering(data, num_centers, iterations=3):
    """Cluster data points using Lloyd's algorithm for center updating."""
    centers = select_simple_centers(data, num_centers)
    
    for _ in range(iterations):
        # Create an empty cluster list for each center
        clusters = [(center, []) for center in centers]
        
        # Assign points to the nearest center
        for point, _ in data:
            distances_to_centers = [compute_distance(point, center) for center in centers]
            closest_center_idx = index_of_min(distances_to_centers)
            clusters[closest_center_idx][1].append(point)
        
        # Recalculate centers as the average of points in each cluster
        centers = []
        for center, cluster_points in clusters:
            if cluster_points:
                x_vals, y_vals = split_coordinates(cluster_points)
                new_center = (average(x_vals), average(y_vals))
            else:
                new_center = center
            centers.append(new_center)
    
    return clusters

# Visualize Clustering Results with Voronoi Diagram
def plot_voronoi_diagram(centers, data_points):
    """Create a Voronoi plot showing centers and data points."""
    voronoi = Voronoi(centers)
    fig = voronoi_plot_2d(voronoi)
    ax = fig.axes[0]
    ax.scatter(*zip(*data_points), color="red", marker="o")
    ax.scatter(*zip(*centers), color="blue", marker="x", label="center")
    return fig, ax

# Generate Gaussian Distributed Data
def generate_gaussian_data(num_centers=10, num_points=10000, std_dev=0.1):
    """Generate data points distributed around multiple Gaussian centers."""
    points = []
    centers = [point for point, _ in generate_uniform_data(0, 1, 0, 1, num_centers)]
    for _ in range(num_points - num_centers):
        chosen_center = centers[random.randint(0, num_centers - 1)]
        points.append(tuple(np.random.normal(loc=chosen_center, scale=std_dev)))
    
    points.extend(centers)
    return points, [(point, 1) for point in points]

gaussian_points, gaussian_data = generate_gaussian_data()

if __name__ == "__main__":
    
    # Q6.1
    visualize_knn_no_transform()
    visualize_knn_with_transform()
    
    # Q6.2
    execute_experiment()
    
    # Q6.16
    # Execute and plot results for uniform data
    simple_centers_uniform = select_simple_centers(uniform_data, 10)
    clustered_uniform_centers = lloyd_clustering(uniform_data, 10)

    # Simple Greedy Heuristic Plot
    fig, ax = plot_voronoi_diagram(simple_centers_uniform, uniform_points)
    ax.set_title("Simple Greedy Heuristic (Uniform Data)")
    fig.legend()

    # Lloyd's Update Plot
    fig, ax = plot_voronoi_diagram([center for center, _ in clustered_uniform_centers], uniform_points)
    ax.set_title("Lloyd's Update (Uniform Data)")
    fig.legend()

    # Query Points Generation
    query_points = generate_uniform_data(0, 1, 0, 1, 10000)

    # Brute Force KNN on Uniform Data
    start_time = time.time()
    for point, _ in query_points:
        knn_classify(uniform_data, point)  # For timing purposes
    brute_force_time = time.time() - start_time
    print(f"Brute Force (Uniform Data): {brute_force_time:.4f} seconds")

    # Branch and Bound KNN on Uniform Data
    start_time = time.time()
    for point, _ in query_points:
        # Find the closest center and use its cluster
        distances_to_centers = [compute_distance(center, point) for center, _ in clustered_uniform_centers]
        nearest_cluster = clustered_uniform_centers[index_of_min(distances_to_centers)][1]
        knn_classify(zip(nearest_cluster, itertools.repeat(100)), point)
    branch_bound_time = time.time() - start_time
    print(f"Branch and Bound (Uniform Data): {branch_bound_time:.4f} seconds")

    # Clustering Gaussian Data
    simple_centers_gaussian = select_simple_centers(gaussian_data, 10)
    clustered_gaussian_centers = lloyd_clustering(gaussian_data, 10)

    # Simple Greedy Heuristic Plot for Gaussian Data
    fig, ax = plot_voronoi_diagram(simple_centers_gaussian, gaussian_points)
    ax.set_title("Simple Greedy Heuristic (Gaussian Data)")
    fig.legend()

    # Lloyd's Update Plot for Gaussian Data
    fig, ax = plot_voronoi_diagram([center for center, _ in clustered_gaussian_centers], gaussian_points)
    ax.set_title("Lloyd's Update (Gaussian Data)")
    fig.legend()

    # Brute Force KNN on Gaussian Data
    start_time = time.time()
    for point, _ in query_points:
        knn_classify(gaussian_data, point)  # For timing purposes
    brute_force_time_gaussian = time.time() - start_time
    print(f"Brute Force (Gaussian Data): {brute_force_time_gaussian:.4f} seconds")

    # Branch and Bound KNN on Gaussian Data
    start_time = time.time()
    for point, _ in query_points:
        # Find the closest center and use its cluster
        distances_to_centers = [compute_distance(center, point) for center, _ in clustered_gaussian_centers]
        nearest_cluster = clustered_gaussian_centers[index_of_min(distances_to_centers)][1]
        knn_classify(zip(nearest_cluster, itertools.repeat(1)), point)
    branch_bound_time_gaussian = time.time() - start_time
    print(f"Branch and Bound (Gaussian Data): {branch_bound_time_gaussian:.4f} seconds")