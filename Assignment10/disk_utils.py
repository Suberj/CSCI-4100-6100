import numpy as np

# Coordinate Class
class Coordinate:
    def __init__(self, x_pos=0.0, y_pos=0.0):
        self.x_pos = x_pos
        self.y_pos = y_pos

    def compute_distance(self, other):
        """Calculate Euclidean distance to another Coordinate."""
        dx = self.x_pos - other.x_pos
        dy = self.y_pos - other.y_pos
        return np.hypot(dx, dy)

    def __getitem__(self, index):
        """Allow access by index: 0 for x, 1 for y."""
        if index == 0: return self.x_pos
        if index == 1: return self.y_pos
        raise IndexError("Coordinate index out of bounds")


# SeparatedDisk Class
class SeparatedDisk:
    def __init__(self, radius, thickness, separation):
        self.radius = radius
        self.thickness = thickness
        self.separation = separation
        self.lower_center = Coordinate(radius * 2 + 1.5 * thickness, thickness + radius)
        self.upper_center = Coordinate(thickness + radius, separation + thickness + radius)
        self.point_list = []
        self.labels = []

    def generate_points(self, total_points):
        """Generate points within the disks with -1/+1 labels."""
        x_min, x_max = 0, (self.thickness + self.radius) * 3 - self.thickness / 2
        y_min, y_max = 0, (self.thickness + self.radius) * 2 + self.separation

        max_distance = self.radius + self.thickness

        # Vectorized point generation
        while len(self.point_list) < total_points:
            random_x = np.random.uniform(x_min, x_max, size=total_points)
            random_y = np.random.uniform(y_min, y_max, size=total_points)

            lower_distances = np.hypot(random_x - self.lower_center.x_pos, random_y - self.lower_center.y_pos)
            upper_distances = np.hypot(random_x - self.upper_center.x_pos, random_y - self.upper_center.y_pos)

            lower_mask = (
                (self.thickness * 0.5 + self.radius <= random_x) &
                (random_x <= x_max) &
                (0 <= random_y) &
                (random_y <= self.lower_center.y_pos) &
                (self.radius <= lower_distances) &
                (lower_distances <= max_distance)
            )

            upper_mask = (
                (0 <= random_x) &
                (random_x <= (self.thickness + self.radius) * 2) &
                (self.radius + self.thickness + self.separation <= random_y) &
                (random_y <= self.upper_center.y_pos + max_distance) &
                (self.radius <= upper_distances) &
                (upper_distances <= max_distance)
            )

            # Add points to the respective disk
            new_points_lower = np.column_stack((random_x[lower_mask], random_y[lower_mask]))
            new_points_upper = np.column_stack((random_x[upper_mask], random_y[upper_mask]))

            self.point_list.extend([Coordinate(x, y) for x, y in new_points_lower])
            self.labels.extend([-1] * len(new_points_lower))

            self.point_list.extend([Coordinate(x, y) for x, y in new_points_upper])
            self.labels.extend([1] * len(new_points_upper))

            # Break if enough points are generated
            if len(self.point_list) >= total_points:
                self.point_list = self.point_list[:total_points]
                self.labels = self.labels[:total_points]
                break

    def plot_points(self, ax):
        """Visualize points with blue (+1) and red (-1) labels."""
        points = np.array([[p.x_pos, p.y_pos] for p in self.point_list])
        colors = ['blue' if label == 1 else 'red' for label in self.labels]
        ax.scatter(points[:, 0], points[:, 1], c=colors, zorder=999)
