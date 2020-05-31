import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from scipy.sparse import csc_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import calibration_metrics as cali
from sklearn.ensemble import GradientBoostingRegressor as GBR
import pickle

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']
feat_lens_dict = {"bibtex": 1836, "ohsumed": 12639, "rcv1": 47236, "tmc": 49060, "wise": 301561, "mscoco": 4096, "cal_yeast": 103}
label_lens_dict = {"bibtex": 159, "ohsumed": 23, "rcv1": 103, "tmc": 22, "wise": 203, "mscoco": 80, "cal_yeast": 14}

dataset = "bibtex_brjava"

(input_val, y_val), (input_test, y_test) = np.load("pickles/{}_probs.p".format(dataset), allow_pickle=True)

print("input_val:", input_val.shape)
print("y_val:", y_val.shape)
print("input_test:", input_test.shape)
print("y_test:", y_test.shape)


def get_prob_feat(probs, threshold=0.5):
    prod = 1.
    for prob in probs:
        if prob < threshold:
            prod *= 1-prob
        else:
            prod *= prob
    return prod

def get_prior_feat(probs, threshold=0.5):
    label_set = []
    for prob in probs:
        if prob < threshold:
            label_set.append(0)
        else:
            label_set.append(1)
    label_set = tuple(label_set)
    if label_set not in priors_dict:
        return 0.
    else:
        return priors_dict[label_set]

def get_card_feat(probs, threshold=0.5):
    cnt = 0
    for prob in probs:
        if prob >= threshold: cnt += 1
    return cnt

def get_label_feat(probs, threshold=0.5):
    label_set = []
    for prob in probs:
        if prob < threshold:
            label_set.append(0)
        else:
            label_set.append(1)
    return label_set

def check_same(probs, ml_labels, threshold=0.5):
    for prob, label in zip(probs, ml_labels):
        if prob < threshold:
            indicator = 0
        else:
            indicator = 1
        if label != indicator:
            return False
    return True

def eval_cali_metrics(pred_probs, multi_labels, gb, thres = 0.5):
    cal_features = []
    cal_raw_preds = []
    cal_labels = []
    for probs, ml in zip(pred_probs, multi_labels):
        prob_feat = get_prob_feat(probs, thres)
        #prior_feat = get_prior_feat(probs, thres) / float(priors_total)
        card_feat = get_card_feat(probs, thres)
        label_feat = get_label_feat(probs, thres)
        feature = [prob_feat, card_feat] + label_feat
        cal_label = check_same(probs, ml, thres)
        
        cal_raw_preds.append(prob_feat)
        cal_features.append(feature)
        cal_labels.append(cal_label)
    cal_features = np.array(cal_features)
    cal_raw_preds = np.array(cal_raw_preds)
    cal_preds = gb.predict(cal_features)
    cal_labels = np.array(cal_labels)
    cal_preds[cal_preds < 0.] = 0.
    cal_preds[cal_preds > 1.] = 1.

    mse, align, sharp, uncert = cali.mse(cal_preds, cal_labels), cali.alignment_error(cal_preds, cal_labels), cali.sharpness(cal_preds, cal_labels), cali.uncertainty(cal_preds, cal_labels)
    raw_mse, raw_align, raw_sharp, raw_uncert = cali.mse(cal_raw_preds, cal_labels), cali.alignment_error(cal_raw_preds, cal_labels), cali.sharpness(cal_raw_preds, cal_labels), cali.uncertainty(cal_raw_preds, cal_labels)
    print("raw_mse:", raw_mse)
    print("raw_align:", raw_align)
    print("raw_sharp:", raw_sharp)
    print("raw_uncertain:", raw_uncert)
    print("uncertain-sharp+align:", raw_uncert-raw_sharp+raw_align)
    print()
    print("mse:", mse)
    print("align:", align)
    print("sharp:", sharp)
    print("uncertain:", uncert)
    print("uncertain-sharp+align:", uncert-sharp+align)

def train_cali_metrics(pred_probs, multi_labels, thres = 0.5):
    cal_features = []
    cal_labels = []
    for probs, ml in zip(pred_probs, multi_labels):
        prob_feat = get_prob_feat(probs, thres)
        #prior_feat = get_prior_feat(probs, thres) / float(priors_total)
        card_feat = get_card_feat(probs, thres)
        label_feat = get_label_feat(probs, thres)
        feature = [prob_feat, card_feat] + label_feat
        cal_label = check_same(probs, ml, 0.5)

        cal_features.append(feature)
        cal_labels.append(cal_label)
    
    gb = GBR(loss='ls', learning_rate=0.1, min_samples_leaf=5, n_estimators=100)
    gb.fit(cal_features, cal_labels)
    return gb

gb = train_cali_metrics(input_val, y_val)
eval_cali_metrics(input_test, y_test, gb)

