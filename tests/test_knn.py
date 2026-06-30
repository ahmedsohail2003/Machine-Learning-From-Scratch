from k_nearest_neighbors import fnKNN, split_dataset, evaluate_knn


def test_fnknn_returns_clear_majority_label():
    # A tiny dataset where the new point sits inside a dense cluster of label 1.
    dataset = [
        [0.0, 0.0, 1],
        [0.1, 0.0, 1],
        [0.0, 0.1, 1],
        [10.0, 10.0, 0],
    ]
    # The three nearest neighbors of [0.05, 0.05] are all label 1.
    assert fnKNN(dataset, [0.05, 0.05], k=3) == 1


def test_split_dataset_is_reproducible():
    dataset = [[float(i), float(i), i % 2] for i in range(10)]

    train_a, test_a = split_dataset(dataset, 0.7, random_seed=123)
    train_b, test_b = split_dataset(dataset, 0.7, random_seed=123)

    # Same seed must yield identical splits.
    assert train_a == train_b
    assert test_a == test_b


def test_split_dataset_train_size():
    dataset = [[float(i), float(i), i % 2] for i in range(10)]

    train, test = split_dataset(dataset, 0.7)

    # Train size is int(n * ratio); test holds the remainder.
    assert len(train) == int(10 * 0.7)
    assert len(test) == 10 - int(10 * 0.7)


def test_split_dataset_loses_no_rows():
    dataset = [[float(i), float(i), i % 2] for i in range(10)]

    train, test = split_dataset(dataset, 0.6)

    # Recombining the split must reproduce the original set of rows.
    recombined = train + test
    assert len(recombined) == len(dataset)
    assert sorted(recombined) == sorted(dataset)


def test_evaluate_knn_perfect_when_train_equals_test():
    dataset = [
        [0.0, 0.0, 0],
        [1.0, 1.0, 1],
        [5.0, 5.0, 0],
        [9.0, 9.0, 1],
    ]
    # With k=1 and train == test, each point is its own nearest neighbor.
    assert evaluate_knn(dataset, dataset, k=1) == 1.0
