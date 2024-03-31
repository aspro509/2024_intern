import numpy as np
from myLib import *

data_dir = "./data/predition/"
predict1 = np.load(data_dir+"prediction1.npy")
label = np.load("./data/face/test_y.npy")
age = np.load("./data/face/Age_test.npy")
sex = np.load("./data/face/Sex_test.npy")

scores_m ,scores_f = group_sex(predict1)
labels_m, labels_f = group_sex(label)
scores_19, scores_20, scores_40, scores_60, scores_20_100 = group_age(predict1)
labels_19, labels_20, labels_40, labels_60, labels_20_100 = group_age(label)

bootstrap_aucs = bootstrap_auc(predict1, label, 100000)
bootstrap_aucs_M = bootstrap_auc(scores_m, labels_m, 100000)
bootstrap_aucs_F = bootstrap_auc(scores_f, labels_f, 100000)

bootstrap_aucs_19 = bootstrap_auc(scores_19, labels_19, 100000)
bootstrap_aucs_20 = bootstrap_auc(scores_20, labels_20, 100000)
bootstrap_aucs_40 = bootstrap_auc(scores_40, labels_40, 100000)
bootstrap_aucs_60 = bootstrap_auc(scores_60, labels_60, 100000)
bootstrap_aucs_20_100 = bootstrap_auc(scores_20_100, labels_20_100, 100000)


arr = [bootstrap_aucs, bootstrap_aucs_M, bootstrap_aucs_F, bootstrap_aucs_19, bootstrap_aucs_20, bootstrap_aucs_40, bootstrap_aucs_60, bootstrap_aucs_20_100]
Group = ["Overall", "male", "female", "Under 20", "20 - 39", "40 - 59", "Over 60", "Over 20"]
for i, group_aucs in enumerate(arr, start=1):
    mean_auc, ci_lower, ci_upper, std_auc = summarize_bootstrap_aucs(group_aucs)
    print(f"model1  {Group[i]}: Mean AUC = {mean_auc:.4f}, Std AUC = {std_auc:.4f}, 95% CI = ({ci_lower:.4f}, {ci_upper:.4f})")