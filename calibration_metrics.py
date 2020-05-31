import numpy as np
import math
from sklearn.calibration import calibration_curve

class Bucket:
    def __init__(self, preds, labels, size=10, minv=0., maxv=1.):
        self.preds = np.array(preds)
        self.labels = np.array(labels)
        self.size = size
        self.minv = minv
        self.maxv = maxv
        self.bucket_len = (maxv-minv)/size

        self.counts = np.zeros(self.size, np.float)
        self.sumProbs = np.zeros(self.size, np.float)
        self.sumLabels = np.zeros(self.size, np.float)
        self.sumSquareProbs = np.zeros(self.size, np.float)
        for prob, label in zip(self.preds, self.labels):
            index = int((prob - self.minv) / self.bucket_len)
            index = max(0, index)
            index = min(self.size-1, index)
            self.counts[index] += 1
            self.sumProbs[index] += prob
            self.sumLabels[index] += label
            self.sumSquareProbs[index] += prob * prob

        print("counts:", self.counts)

        self.accs = np.zeros(self.size, np.float)
        self.aveProbs = np.zeros(self.size, np.float)
        for i in range(self.size):
            if self.counts[i]:
                self.accs[i] = self.sumLabels[i] / self.counts[i]
                self.aveProbs[i] = self.sumProbs[i] / self.counts[i]

        self.total_counts = np.sum(self.counts)
        self.total_labels = np.sum(self.sumLabels)

def mse(preds, labels):
    n_preds, n_labels = len(preds), len(labels)
    assert n_preds == n_labels
    return np.mean(np.square(preds-labels))

def sharpness(preds, labels):
    bucket = Bucket(preds, labels)
    avg = bucket.total_labels / bucket.total_counts
    sum = 0.
    for i in range(bucket.size):
        if bucket.counts[i]:
            sum += bucket.counts[i] * math.pow(bucket.accs[i]-avg, 2)
    return sum/(bucket.total_counts-1)

def alignment_error(preds, labels):
    bucket = Bucket(preds, labels)
    sum = 0.
    sum1 = 0.
    for i in range(bucket.size):
        if bucket.counts[i]:
            sum1 += math.pow(bucket.accs[i], 2) * bucket.counts[i] - 2 * bucket.sumProbs[i] * bucket.accs[i] + bucket.sumSquareProbs[i]
            sum += ((bucket.accs[i]-bucket.aveProbs[i])**2)*bucket.counts[i]
    return sum/bucket.total_counts

def uncertainty(preds, labels):
    sum = float(np.sum(labels))
    count = len(labels)
    sumSquare = np.sum(np.square(labels))
    return (sumSquare-sum*sum/count)/(count-1.)


def _bucket_sizes(p, n_bins=10):
    lengths = list()
    iv_size = 1./n_bins
    for i in range(n_bins):
        l = len([p_j for p_j in p if i*iv_size <= p_j <= (i+1)*iv_size])
        if l:
            lengths.append(l)
    return lengths

def volo_calibration_error(proba, y):
    p_true, p_emp = calibration_curve(y, proba, n_bins=10)
    L = _bucket_sizes(proba)
    return sum([l*(pe - pt)**2 for l, pe, pt in zip(L, p_emp, p_true)]) / sum(L)
def volo_sharpness(proba, y):
    p_true, p_emp = calibration_curve(y, proba, n_bins=10)
    L = _bucket_sizes(proba)
    y_avg = np.mean(y)
    return sum([l*(pt - y_avg)**2 for l, pt in zip(L, p_true)]) / (sum(L)-1)
def volo_brier_score(proba, y):
    return sum([(p_i - y_i)**2 for p_i, y_i in zip(proba, y)]) / len(y)
def volo_accuracy(proba, y):
    y_pred = [1 if p_i >= 0.5 else 0 for p_i in proba]
    return sum([1.0 if y_pred_i == y_i else 0.0 for y_pred_i, y_i in zip(y_pred, y)]) / len(y)
