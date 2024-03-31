import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit


data_dir = "./data/face/"
def load_data():
    global X, y, train_age, X_224, X_112, X_56

    X = np.load(data_dir+"train_X.npy")
    train_age = np.load(data_dir+"Age_train.npy")
    X_224 = X[:,:,:,-1]
    X_112 = np.load(data_dir+"train_X_112.npy")
    X_56 = np.load(data_dir+"train_X_56.npy")
    y = np.load(data_dir+"train_y.npy")
load_data()
N = len(y)

indices = np.arange(len(X))
np.random.seed(123)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
age = train_age[indices]

combined_data = np.column_stack((y, train_age))

stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2,  random_state=123)

# Perform stratified split
for train_index, val_index in stratified_splitter.split(combined_data, combined_data[:, 0]):
    train_img, valid_img = X[train_index], X[val_index]
    train_labels, valid_labels = y[train_index], y[val_index]

t_total  = len(train_labels)
t_P = train_labels.sum()
t_N = t_total - t_P 

v_total = len(valid_labels)
v_P = valid_labels.sum()
v_N = v_total - v_P

print("X_train shape:", train_img.shape)
print("y_train shape:", train_labels.shape)
print("X_val shape:", valid_img.shape)
print("y_val shape:", valid_labels.shape)
print("train P:{:.4f}, N:{:.4f}, total:{}".format(t_P/t_total, t_N/t_total, t_total))
print("valid P:{:.4f}, N:{:.4f}, total:{}".format(v_P/v_total, v_N/v_total, v_total))
