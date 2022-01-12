import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import csv
from liblinear.commonutil import evaluations
from libsvm.svmutil import *

def load_data(path):
    df = pd.read_csv(path)
    x, y, id = [], [], []
    
    return x, y

def save_pred(pred, labels, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Customer ID', 'Churn Category'])
        for i in range(len(pred)):
            writer.writerow([dataID[i][:12], pred[i]])


def xgb():
    x_train, y_train = load_data("./feature/train.dat")
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
    
