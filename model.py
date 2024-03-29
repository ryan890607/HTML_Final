import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import csv
from libsvm.svmutil import *
import random
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from liblinear.liblinearutil import *
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

def logistic(trainData, label, test):
    # set param
    trainData, validData, label, validLabel = train_test_split(trainData, label, test_size=0.2, random_state=12)
    #train
    model = train(label, trainData, '-s 0 -c 10 -e 0.01')
    trash, acc, val = predict(validLabel, validData, model)
    print('val score = ', acc[0])
    predicted, acc, val = predict([], test, model)
    return predicted

def boostAggregate(trainPool, labelPool, test):
    result = [[0, 0, 0, 0, 0, 0] for i in range(len(test))]
    for i in range(15):
        model = XGBClassifier(n_estimators=100, max_depth = 6, learning_rate= 0.15, objective='multi:softmax', booster='gbtree', gamma=0.1, min_child_weight=3, num_class=6)
        model.fit(trainPool[i], labelPool[i])
        #print(train.shape)
        predicted = model.predict(test)
        for j in range(len(test)):
            result[j][int(predicted[j])] += 1    
    predicted = []
    for aggregate in result:
        predicted.append(aggregate.index(max(aggregate)))
    return predicted

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

    train, validData, label, validLabel = train_test_split(train, label, test_size=0.2, random_state=12)

    trainPool, labelPool = [], []
    for i in range(15):
        curTrain, curLabel = [], []
        for j in range(1740):
            index = random.randrange(len(train))
            curTrain.append(train[index].tolist())
            curLabel.append(label[index])
        print(len(curTrain[1]))
        print(len(curLabel))
        trainPool.append(curTrain)
        labelPool.append(curLabel)

    predicted = boostAggregate(trainPool, labelPool, test)
    validated = boostAggregate(trainPool, labelPool, validData)
    score = 0
    for i in range(len(validated)):
        if validated[i] == validLabel[i]:
            score += 1
    score /= len(validated)
    print('score = ', score)
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
print(f"element_cnt: {element_cnt}")
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
# print(x_train_nan)
# print(x_test_nan)
# print(y_train_balance)
need_delete = []
for i, v in enumerate(y_train):
    if np.isnan(v):
        need_delete.append(i)

x_train = np.delete(x_train, need_delete, axis = 0)
y_train = np.delete(y_train, need_delete, axis = 0)
#print(x_train)
#print(y_train)
#print(x_test)
prediction = boost(x_train_balance, y_train_balance, x_test)
#prediction = logistic(x_train, y_train, x_test)
save_pred(prediction, "predict.csv")

#tmp1 = [i for i in range(6) for j in range(200)]
#tmp2 = [i for i in range(2) for j in range(600)]
#print(tmp1, tmp2)
#print(f1_score(tmp1, tmp2, average='micro'))
