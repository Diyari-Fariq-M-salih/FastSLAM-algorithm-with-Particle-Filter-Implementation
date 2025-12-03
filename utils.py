def map_accuracy(estimated, ground_truth):
    est = estimated > 0     # occupied
    gt  = ground_truth > 0.6

    tp = np.sum(est & gt)
    tn = np.sum(~est & ~gt)
    fp = np.sum(est & ~gt)
    fn = np.sum(~est & gt)

    return (tp + tn) / (tp + tn + fp + fn)
