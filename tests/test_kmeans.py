import numpy as np
import pytest

from k_means_clustering import euclidean_distance, k_means


def test_euclidean_distance_3_4_5():
    # Classic 3-4-5 right triangle: distance is exactly 5.0.
    assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0


def test_k_means_separates_two_clusters():
    data = [[0, 0], [0.1, 0], [10, 10], [10.1, 10]]

    cluster_assignments, centroids = k_means(data, k=2)

    # Points 0 and 1 belong together; points 2 and 3 belong together.
    assert cluster_assignments[0] == cluster_assignments[1]
    assert cluster_assignments[2] == cluster_assignments[3]
    assert cluster_assignments[0] != cluster_assignments[2]

    # The two centroids are the means of each tight pair of points.
    centroid_by_label = {
        int(cluster_assignments[0]): centroids[cluster_assignments[0]],
        int(cluster_assignments[2]): centroids[cluster_assignments[2]],
    }
    assert centroid_by_label[int(cluster_assignments[0])] == pytest.approx([0.05, 0.0])
    assert centroid_by_label[int(cluster_assignments[2])] == pytest.approx([10.05, 10.0])
