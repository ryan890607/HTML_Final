import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import csv
from libsvm.svmutil import *
import random
from tqdm import tqdm
# from liblinear.commonutil import evaluations

test_IDs = pd.read_csv("data/Test_IDs.csv")
testID_list = list(test_IDs["Customer ID"])

def load_data(path, type):
    df = pd.read_csv(path)
    # print(df.info)
    datas = df.values
    datas = datas[:, 1:]
    if(type == 0): 
        x, y = datas[:, :-1], datas[:, -1]
        return x, y
    else:
        return datas

def save_pred(prediction, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i in range(len(prediction)):
            writer.writerow([testID_list[i], int(prediction[i])])


def C_svm(train, label, test):
    # set param
    param = svm_parameter('-s 0 -t 2 -c 1')
    #train
    prob  = svm_problem(label, train)
    m = svm_train(prob, param)
    label, acc, val = svm_predict(label, train, m)
    E_in = (1 - acc[0] / 100)
    print(E_in)
    label, acc, val = svm_predict([], test, m)
    return label


# Usage
x_train, y_train = load_data("./features/train.csv", 0)
print(x_train.shape, y_train.shape)
x_train_drop, y_train_drop = [], []
# drop data without label
for i, j in zip(x_train, y_train):
    if not np.isnan(j):
        x_train_drop.append(i)
        y_train_drop.append(j)
x_train_drop, y_train_drop = np.array(x_train_drop), np.array(y_train_drop)
print(x_train_drop.shape, y_train_drop.shape)
element_cnt = dict((a, y_train_drop.tolist().count(a)) for a in y_train_drop)
print(f"element_cnt: {element_cnt}")
# remove some 0 label
x_train_balance, y_train_balance = [], []
for i, j in zip(x_train_drop, y_train_drop):
    if j != 0 or random.random() > 0.85:
        x_train_balance.append(i)
        y_train_balance.append(j)
x_train_balance, y_train_balance = np.array(x_train_balance), np.array(y_train_balance)
print(x_train_balance.shape, y_train_balance.shape)
element_cnt = dict((a, y_train_balance.tolist().count(a)) for a in y_train_balance)
print(f"element_cnt: {element_cnt}")
# test
x_test = load_data("./features/test.csv", 1)
print(x_test.shape)
x_train_dict, x_test_dict = [], []
# trainsform to dict
for i in tqdm(range(len(x_train_balance))): x_train_dict.append({j: x_train_balance[i][j] for j in range(len(x_train_balance[i]))})
for i in tqdm(range(len(x_test))): x_test_dict.append({j: x_test[i][j] for j in range(len(x_test[i]))})
# remove nan data
x_train_nan, x_test_nan = [], []
def filter(data):
    return {k: v for k, v in data.items() if not np.isnan(v)}
for i in tqdm(range(len(x_train_balance))): x_train_nan.append(filter(x_train_dict[i]))
for i in tqdm(range(len(x_test))): x_test_nan.append(filter(x_test_dict[i]))
print(x_train_nan[0])
# train
prediction = C_svm(x_train_nan, y_train_balance, x_test_nan)
save_pred(prediction, "predict.csv")

