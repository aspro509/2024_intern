from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn


class evaluate:

  def __init__(self, labels, predict, val_labels, val_predict, sex, age):
    self.labels = labels.squeeze()
    self.predict = predict.squeeze()
    self.val_labels = val_labels.squeeze()
    self.val_predict = val_predict.squeeze()
    self.figsize = (10, 10)
    self.threshold = 0.5
    self.sex = sex
    self.age = age

  def set_threshold(self):
    # calculate Youden’s index in valid set
    val_label = self.val_labels
    val_predict = self.val_predict
    def calculate_Y_index(threshold):
      TP = np.logical_and(val_label > threshold, val_predict >= threshold).sum()
      TN = np.logical_and(val_label < threshold, val_predict <  threshold).sum()
      FP = np.logical_and(val_label < threshold, val_predict >= threshold).sum()
      FN = np.logical_and(val_label > threshold, val_predict <  threshold).sum()

      try:
        assert TP + FN == val_label.sum()
        assert TN + FP == val_label.shape[0] - val_label.sum()
      except:
        print("error TP + FN != Positive or TN + FP != Negative")
      Sensitivity = TP / (TP + FN)
      Specificity = TN / (FP + TN)
      return Sensitivity + Specificity
    
    thresholds = val_predict[1:-1]
    y = np.vectorize(calculate_Y_index)(thresholds)
    self.threshold = thresholds[np.argmax(y)]


  def result(self):
    threshold = self.threshold
    acc = accuracy_score(self.labels, self.predict>=threshold)
    auc = roc_auc_score(self.labels, self.predict)

    TP = np.logical_and(self.labels > threshold, self.predict >= threshold).sum()
    TN = np.logical_and(self.labels < threshold, self.predict < threshold).sum()
    FP = np.logical_and(self.labels < threshold, self.predict >= threshold).sum()
    FN = np.logical_and(self.labels > threshold, self.predict < threshold).sum()

    assert TP + FN == self.labels.sum()
    assert TN + FP == self.labels.shape[0] - self.labels.sum()

    Sensitivity = round(TP / (TP + FN), 4)
    Specificity = round(TN / (FP + TN), 4)
    if TP + FP == 0 or TN + FN == 0:
      PPV = None
      NPV = None
    else:
      PPV = round(TP / (TP + FP), 4)
      NPV = round(TN / (TN + FN), 4)

    return (Sensitivity, Specificity, PPV, NPV, acc, auc)
  
  def plot_save_roc(self, name, filename, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(self.labels, self.predict)
    plt.figure(figsize=self.figsize)
    plt.plot(100*fp, 100*tp, label=name, linewidth=4, **kwargs)
    plt.xlabel('False positives [%]', fontsize=20)
    plt.ylabel('True positives [%]', fontsize=20)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.plot([0,100], [0,100], linestyle = '--', color = 'k')
    ax = plt.gca()
    ax.set_aspect(aspect=1)
    plt.legend()
    plt.savefig(filename)

  def plot_cm(self, filename, **kwargs):
      
    cm = confusion_matrix(self.labels, self.predict > self.threshold)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d",**kwargs)

    plt.ylabel('Actual Values', fontsize=20)
    plt.xlabel('Predicted Values', fontsize=20)

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    plt.savefig(filename)
