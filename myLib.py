import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def group_sex(arr, sex):

    predict_M = arr[sex>0.5]
    predict_F = arr[sex<0.5]

    return predict_M, predict_F

def group_age(arr, age):
    predict_0_19 =  arr[age<20]
    predict_20_39 = arr[np.logical_and(age>=20, age <40)]
    predict_40_59 = arr[np.logical_and(age>=40, age <60)]
    predict_60 = arr[age>=60]
    predict_20 = arr[age>=20]

    return predict_0_19, predict_20_39, predict_40_59, predict_60, predict_20

def bootstrap_auc(scores, labels, n_bootstraps=10000):
    bootstrap_aucs = []

    for i in range(n_bootstraps):
        if (i+1) % (n_bootstraps//10) ==0:
            print(f"{(i+1)/n_bootstraps*100}%, ", end="")
        # Resample scores and labels with replacement
        indices = resample(np.arange(len(labels)))
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        # Calculate AUC for the bootstrap sample
        auc = roc_auc_score(boot_labels, boot_scores)
        bootstrap_aucs.append(auc)
    print()

    return np.array(bootstrap_aucs)

def summarize_bootstrap_aucs(bootstrap_aucs):
    mean_auc = np.mean(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    return mean_auc, ci_lower, ci_upper