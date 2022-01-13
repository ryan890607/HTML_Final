import pandas as pd
import numpy as np
import xgboost as xgb
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
    param = svm_parameter('-s 0 -t 1 -d 5 -g 2 -r 1 -c 10 -q')
    #train
    prob  = svm_problem(label, train)
    m = svm_train(prob, param)
    label, acc, val = svm_predict(label, train, m)
    E_in = (1 - acc[0] / 100)
    print(E_in)
    label, acc, val = svm_predict([], test, m)
    return label

def boost(train, label, test):
    #param = {
    #    'booster': 'gbtree',
    #    'objective': 'multi:softmax',
    #    'num_class': 6,
    #    'gamma': 0.1,                  
    #    'max_depth': 12,               
    #    'lambda': 2,
    #    'eta': 0.007,
    #    'seed': 1000,
    #}
    #plst = param.items()

    model = XGBClassifier(n_estimators=100, max_depth = 6, learning_rate= 0.2, objective='multi:softmax', booster='gbtree')
    model.fit(train, label)
    #print(train.shape)
    predicted = model.predict(test)
    print('score = ', model.score(train, label))
    return predicted
    #save_pred(predicted, 'predict.csv')

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
# print(x_train_drop.shape, y_train_drop.shape)
element_cnt = dict((a, y_train_drop.tolist().count(a)) for a in y_train_drop)
# print(f"element_cnt: {element_cnt}")
# remove some 0 label (balance data)
x_train_balance, y_train_balance = [], []
for i, j in zip(x_train_drop, y_train_drop):
    if j != 0 or random.random() > 0.8:
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
# print(x_train_nan[0])
# train
#prediction = C_svm(x_train_nan, y_train_balance, x_test_nan)
#x_train_nan = np.array(x_train_nan)
#x_test_nan = np.array(x_test_nan)
print(x_train_nan)
print(x_test_nan)
print(y_train_balance)
need_delete = []
for i, v in enumerate(y_train):
    if np.isnan(v):
        need_delete.append(i)

x_train = np.delete(x_train, need_delete, axis = 0)
y_train = np.delete(y_train, need_delete, axis = 0)
#print(x_train)
#print(y_train)
#print(x_test)
prediction = boost(x_train, y_train, x_test)
save_pred(prediction, "predict.csv")

