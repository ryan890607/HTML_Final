import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import csv
# from liblinear.commonutil import evaluations
# from libsvm.svmutil import *

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


# Usage
x, y = load_data("./features/train.csv", 0)
print(x.shape, y.shape)
x = load_data("./features/test.csv", 1)
print(x.shape)






def save_pred(prediction, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i in range(len(prediction)):
            writer.writerow([testID_list[i], prediction[i]])


def xgb():
    x_train, y_train = load_data("./feature/train.csv")
    model = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
    model.fit(x_train, y_train)
    predicted = model.predict(x_train)

def C_svm():
    y, x = svm_read_problem('./feature/train.dat')
    # relabel
    # set param
    param = svm_parameter('-s 0 -t 1 -d 3 -g 1 -r 1 -c 10')
    #train
    E_in = []
    for i in range(5):
        prob  = svm_problem(y[i], x)
        m = svm_train(prob, param)
        label, acc, val = svm_predict(y[i], x, m)
        E_in.append(1 - acc[0] / 100)
    # E_in
    print(E_in)
    print(np.argmax(E_in) + 2)
    
