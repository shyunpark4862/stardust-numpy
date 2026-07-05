"""Integration-style workflows combining many ndarray features."""

from __future__ import annotations

import math

from numpy import *  # noqa: F403

NAN = float("nan")
INF = float("inf")


def _close(values, expected, *, rel_tol=1e-9, abs_tol=1e-9):
    if isinstance(values, (int, float)):
        values = [values]
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(expected, (int, float)):
        expected = [expected]
    elif hasattr(expected, "tolist"):
        expected = expected.tolist()
    return all(
        math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
        for left, right in zip(values, expected)
    )


def test_softmax_classification_scores():
    """Stable softmax over class logits for a mini batch."""
    logits = array(
        [
            [1.0, 2.0, 0.5],
            [0.0, -1.0, 3.0],
        ]
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    assert probabilities.shape == (2, 3)
    assert _close(probabilities.sum(axis=1).tolist(), [1.0, 1.0])
    assert probabilities[0].tolist()[1] > probabilities[0].tolist()[0]
    assert probabilities[1].tolist()[2] > probabilities[1].tolist()[0]


def test_feature_matrix_zscore_and_clip():
    """Normalize tabular features, then clip extreme z-scores."""
    features = array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / features.std(axis=0, keepdims=True)
    clipped = clip(scaled, -2.0, 2.0)

    assert _close(scaled.mean(axis=0).tolist(), [0.0, 0.0, 0.0])
    assert clipped.min() >= -2.0
    assert clipped.max() <= 2.0
    assert clipped.shape == features.shape


def test_linear_model_forward_pass_with_bias():
    """Build an augmented design matrix and run a batched forward pass."""
    samples = array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]
    )
    bias = ones((samples.shape[0], 1))
    design = concatenate([bias, samples], axis=1)
    weights = array([0.5, 1.0, -2.0])

    predictions = design.dot(weights)

    assert design.shape == (3, 3)
    assert predictions.shape == (3,)
    assert _close(predictions.tolist(), [-2.5, 0.5, -4.5])


def test_covariance_matrix_from_centered_features():
    """Compute a sample covariance matrix with transpose and dot."""
    data = array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )
    centered = data - data.mean(axis=0, keepdims=True)
    covariance = centered.T.dot(centered) / (data.shape[0] - 1)

    assert covariance.shape == (3, 3)
    assert math.isclose(covariance[0, 0], 1.6666666666666667)
    assert math.isclose(covariance[0, 1], covariance[1, 0])
    assert trace(covariance) > 0.0


def test_nan_imputation_and_valid_mean_recompute():
    """Impute missing sensor readings, then recompute a robust summary."""
    readings = array([1.0, NAN, 3.0, NAN, 5.0, 7.0])
    missing = isnan(readings)
    valid_count = logical_not(missing).sum()
    valid_sum = where(missing, 0.0, readings).sum()
    fill_value = valid_sum / valid_count
    cleaned = where(missing, fill_value, readings)

    assert valid_count == 4
    assert _close(cleaned.tolist(), [1.0, 4.0, 3.0, 4.0, 5.0, 7.0])
    assert math.isclose(cleaned.mean(), 4.0)
    assert isnan(cleaned).sum() == 0


def test_image_batch_spatial_mean_subtraction():
    """Subtract per-image spatial means from a small batch tensor."""
    batch = arange(36, dtype=float).reshape(4, 3, 3)
    spatial_means = batch.mean(axis=(1, 2), keepdims=True)
    normalized = batch - spatial_means

    assert spatial_means.shape == (4, 1, 1)
    assert _close(normalized.mean(axis=(1, 2)).tolist(), [0.0, 0.0, 0.0, 0.0])
    assert normalized.shape == batch.shape


def test_confusion_matrix_style_label_counting():
    """Count predicted-vs-true label pairs using unique inverse indices."""
    y_true = array([0, 1, 2, 1, 0, 2, 2])
    y_pred = array([0, 2, 2, 1, 0, 1, 2])
    labels, true_inverse = unique(y_true, return_inverse=True)
    num_classes = labels.size

    counts = zeros((num_classes, num_classes), dtype=int)
    for index in range(y_true.size):
        counts[true_inverse[index], y_pred[index]] += 1

    assert labels.tolist() == [0, 1, 2]
    assert counts.tolist() == [
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 2],
    ]
    assert counts.sum() == y_true.size


def test_top_k_selection_with_sort_and_slice():
    """Select top-2 scores per row using argsort, sort, and slicing."""
    scores = array(
        [
            [3.0, 1.0, 4.0, 2.0],
            [8.0, 6.0, 7.0, 5.0],
        ]
    )
    top_two_values = sort(scores, axis=1)[:, -2:]

    assert top_two_values[0].tolist() == [3.0, 4.0]
    assert top_two_values[1].tolist() == [7.0, 8.0]


def test_weighted_moving_average_with_cumsum():
    """Compute a cumulative moving average for a short sales series."""
    sales = array([10.0, 20.0, 30.0, 40.0, 50.0])
    cumulative = sales.cumsum()
    counts = arange(1, sales.size + 1, dtype=float)
    moving_average = cumulative / counts

    assert _close(moving_average.tolist(), [10.0, 15.0, 20.0, 25.0, 30.0])
    assert moving_average[-1] == sales.mean()


def test_complex_spectrum_magnitude_pipeline():
    """Convert complex frequency bins to magnitudes and phases."""
    spectrum = array([1 + 1j, 0 + 2j, -3 + 0j, 4 - 4j])
    magnitudes = absolute(spectrum)
    phases = angle(spectrum)
    power = square(magnitudes)

    assert _close(
        magnitudes.tolist(),
        [math.sqrt(2), 2.0, 3.0, math.sqrt(32)],
    )
    assert phases.shape == spectrum.shape
    assert _close(power.tolist(), [2.0, 4.0, 9.0, 32.0])
    assert real(conjugate(spectrum)).tolist() == spectrum.real.tolist()


def test_broadcasted_masking_and_boolean_filtering():
    """Filter valid measurements with broadcasted thresholds and nonzero."""
    measurements = array(
        [
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0],
        ]
    )
    lower = array([[2.0, 4.0, 8.0]])
    upper = array([[8.0], [7.0], [12.0]])
    valid = logical_and(measurements >= lower, measurements <= upper)
    filtered = measurements[valid]
    rows, cols = nonzero(valid)

    assert filtered.tolist() == [5.0, 2.0, 6.0, 3.0, 7.0, 11.0]
    assert rows.tolist() == [0, 1, 1, 2, 2, 2]
    assert cols.tolist() == [1, 0, 1, 0, 1, 2]


def test_matrix_chain_with_reshape_transpose_and_matmul():
    """Reshape flat data, transpose, and multiply through a small chain."""
    left = arange(24, dtype=float).reshape(4, 6)
    kernel = array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projected = left.dot(kernel)
    transposed = projected.T
    combined = transposed.matmul(diag(array([1.0, 2.0, 3.0, 4.0])))

    assert projected.shape == (4, 3)
    assert combined.shape == (3, 4)
    assert (
        combined.sum()
        == projected.T.dot(diag(array([1.0, 2.0, 3.0, 4.0]))).sum()
    )


def test_nditer_driven_custom_reduction():
    """Use nditer to compute a dot-product-like score without calling dot()."""
    left = array([1.0, 2.0, 3.0, 4.0])
    right = array([4.0, 3.0, 2.0, 1.0])
    total = 0.0

    for left_value, right_value in nditer([left, right]):
        total += left_value * right_value

    assert total == left.dot(right)


def test_stack_concatenate_and_unique_pipeline():
    """Merge train/validation splits, then inspect unique labels."""
    train = array([0, 1, 1, 2, 2, 2])
    validation = array([2, 0, 1])
    batched = concatenate([train, validation])
    stacked = stack([train[:3], validation])
    labels, counts = unique(batched, return_counts=True)

    assert batched.tolist() == [0, 1, 1, 2, 2, 2, 2, 0, 1]
    assert stacked.shape == (2, 3)
    assert labels.tolist() == [0, 1, 2]
    assert counts.tolist() == [2, 3, 4]


def test_masked_update_with_where_and_out():
    """Apply a bounded in-place update using where and ufunc out."""
    values = array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = array([True, False, True, False, True])
    out = zeros(values.shape, dtype=float)
    updated = add(values, 10.0, where=mask, out=out)

    assert updated is out
    assert updated.tolist() == [11.0, 0.0, 13.0, 0.0, 15.0]
    assert updated[1] == 0.0
    assert updated[3] == 0.0


def test_sales_anomaly_detection_pipeline():
    """Detect anomalies with rolling mean, threshold mask, and clip."""
    daily_sales = array([100.0, 105.0, 98.0, 250.0, 102.0, 99.0, 101.0])
    baseline = concatenate(
        [daily_sales[:1], daily_sales.cumsum()[:-1]]
    ) / arange(1, daily_sales.size + 1, dtype=float)
    deviation = absolute(daily_sales - baseline)
    threshold = deviation.mean() + 2.0 * deviation.std()
    anomalies = deviation > threshold
    cap = baseline + threshold
    capped = where(daily_sales > cap, cap, daily_sales)
    capped = where(capped < 0.0, 0.0, capped)
    cleaned = where(anomalies, baseline, capped)

    assert anomalies.tolist().count(True) >= 1
    assert cleaned[3] < daily_sales[3]
    assert cleaned.sum() < daily_sales.sum()
    assert nonzero(anomalies)[0].tolist() == [3]


def test_mini_neural_layer_with_outer_and_reductions():
    """Combine outer products, reductions, and nonlinearities for a tiny layer."""
    inputs = array([1.0, 2.0, 3.0])
    weights = array([0.5, -1.0, 0.25])
    bias = array([0.1, -0.2, 0.3])
    pairwise = outer(inputs, weights)
    activated = where(pairwise >= 0.0, pairwise, 0.0)
    logits = activated.sum(axis=0) + bias
    probabilities = exp(logits - logits.max())
    probabilities = probabilities / probabilities.sum()

    assert pairwise.shape == (3, 3)
    assert activated.min() >= 0.0
    assert _close(probabilities.sum(), 1.0)
    assert probabilities.max() <= 1.0


def test_time_window_features_with_stride_like_indexing():
    """Build overlapping windows with reshape, transpose, and axis reductions."""
    series = arange(12, dtype=float)
    windows = stack(
        [
            series[0:4],
            series[2:6],
            series[4:8],
            series[6:10],
        ]
    )
    normalized = windows - windows.mean(axis=1, keepdims=True)
    features = concatenate(
        [
            normalized.max(axis=1, keepdims=True),
            normalized.min(axis=1, keepdims=True),
            normalized.std(axis=1, keepdims=True),
        ],
        axis=1,
    )

    assert windows.shape == (4, 4)
    assert _close(normalized.mean(axis=1).tolist(), [0.0, 0.0, 0.0, 0.0])
    assert features.shape == (4, 3)
    assert features.max() >= features.min()


def test_large_batch_statistics_remain_stable():
    """Exercise large-ish tensors across creation, reshape, and reductions."""
    batch = arange(5000, dtype=float).reshape(50, 10, 10)
    batch_mean = batch.mean()
    per_sample_mean = batch.mean(axis=(1, 2))
    centered = batch - per_sample_mean.reshape(50, 1, 1)
    rms = math.sqrt((centered * centered).mean())

    assert batch.shape == (50, 10, 10)
    assert math.isclose(batch_mean, per_sample_mean.mean())
    assert _close(centered.mean(axis=(1, 2)).tolist(), [0.0] * 50)
    assert rms > 0.0
    assert str(batch).splitlines()
