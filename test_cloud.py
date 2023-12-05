import numpy as np
import pytest

from cloud import Point, PointCloud  # Replace with your actual module name


class TestPointCloud:
    @pytest.fixture
    def sample_point_cloud(self):
        points = [
            Point(np.array([x, y]), x**2 + y**2)
            for x in range(-2, 3)
            for y in range(-2, 3)
        ]
        return PointCloud(points)

    def test_compute_gradient_at_point(self, sample_point_cloud):
        # Choose a point and test the gradient calculation
        point_index = 12  # Index of point (0,0) in the grid
        expected_gradient = np.array(
            [0, 0]
        )  # The gradient at (0,0) should be (0,0) for the function x^2 + y^2
        calculated_gradient = sample_point_cloud.compute_gradient_at_point(point_index)
        np.testing.assert_array_almost_equal(
            calculated_gradient, expected_gradient, decimal=5
        )

    def test_compute_surface_normals(self, sample_point_cloud):
        # Test the surface normal calculation for all points
        surface_normals = sample_point_cloud.compute_surface_normals()
        assert len(surface_normals) == len(sample_point_cloud.points)
        for normal in surface_normals:
            assert len(normal) == 2  # Ensure each normal is a 2D vector


# Run the tests
pytest.main()
