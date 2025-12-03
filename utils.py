import numpy as np

def map_accuracy(est, gt):
    binary_est = est > 0.5
    binary_gt = gt > 0.6

    tp = np.sum(binary_est & binary_gt)
    tn = np.sum(~binary_est & ~binary_gt)
    fp = np.sum(binary_est & ~binary_gt)
    fn = np.sum(~binary_est & binary_gt)

    return (tp + tn) / (tp + tn + fp + fn + 1e-6)
