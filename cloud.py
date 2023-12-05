from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
from scipy.spatial import KDTree


@dataclass
class Point:
    coordinates: np.ndarray
    value: float = 0.0  # Scalar value of the point

    def __str__(self):
        return f"Point(coordinates={self.coordinates}, value={self.value})"


@dataclass
class PointCloud:
    points: List[Point]
    kd_tree: KDTree = field(init=False, default=None)

    def __post_init__(self):
        self._update_kd_tree()

    def add_point(self, point: Point):
        self.points.append(point)
        self._update_kd_tree()

    def remove_point(self, point: Point):
        self.points.remove(point)
        self._update_kd_tree()

    def _update_kd_tree(self):
        self.kd_tree = KDTree([point.coordinates for point in self.points])

    def find_nearest_neighbors(self, point: Point, k: int) -> List[Point]:
        _, indices = self.kd_tree.query(point.coordinates, k)
        return [self.points[i] for i in indices]

    def compute_gradient_at_point(
        self, point_index: int, h: float = 1e-5
    ) -> np.ndarray:
        point = self.points[point_index]
        gradient = np.zeros_like(point.coordinates)

        for i in range(len(point.coordinates)):
            forward_point = np.copy(point.coordinates)
            backward_point = np.copy(point.coordinates)
            forward_point[i] += h
            backward_point[i] -= h

            forward_index = self.kd_tree.query(forward_point)[1]
            backward_index = self.kd_tree.query(backward_point)[1]

            f_plus = self.points[forward_index].value
            f_minus = self.points[backward_index].value
            gradient[i] = (f_plus - f_minus) / (2 * h)

        return gradient

    def compute_surface_normals(self) -> List[np.ndarray]:
        return [self.compute_gradient_at_point(i) for i in range(len(self.points))]
