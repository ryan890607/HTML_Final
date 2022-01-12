import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from scipy import stats

demographics = pd.read_csv("data/demographics.csv")
location = pd.read_csv("data/location.csv")
population = pd.read_csv("data/population.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")
satisfaction = pd.read_csv("data/satisfaction.csv")
services = pd.read_csv("data/services.csv")
status = pd.read_csv("data/status.csv")
train_IDs = pd.read_csv("data/Train_IDs.csv")
test_IDs = pd.read_csv("data/Test_IDs.csv")

trainID_list, testID_list = list(train_IDs["Customer ID"]), list(test_IDs["Customer ID"])
# print(trainID_list, testID_list)
train_feature, test_feature = [[] for i in range(len(trainID_list))], [[] for i in range(len(testID_list))]


# For those df which has userID
def add_feature(x, y, df):
    # add data not in df but in train or test
    for id in tqdm(trainID_list):
        # print(df["Customer ID"][0], id)
        if id not in list(df["Customer ID"]): 
            tmp = [id]
            tmp.extend([np.nan for i in range(len(df.columns) - 1)])
            df.loc[len(df)] = tmp
    for id in tqdm(testID_list):
        if id not in list(df["Customer ID"]): 
            tmp = [id]
            tmp.extend([np.nan for i in range(len(df.columns) - 1)])
            df.loc[len(df)] = tmp
    # for discrete col
    str_col = []
    for column_name in df.columns:
        if column_name == "Customer ID": continue
        # print(column_name)
        for index in range(len(df)):
            if not df.isnull()[column_name][index]: break
        # discrete
        if isinstance(df.loc[index][column_name], str): 
            # print(df.mode().iloc[0])
            df[column_name] = df[column_name].fillna(df.mode().loc[0][column_name])
            str_col.append(column_name)
        # continuous
        else: 
            # print(type(df.loc[index][column_name]), column_name, df.loc[index][column_name], index)
            df[column_name] = df[column_name].fillna(df[column_name].mean())
    
    # one hot encoding
    df = pd.get_dummies(df, columns=str_col)
    # print(df.head(5))
    # train test split
    train, test = [], []
    for index in tqdm(range(len(df))):
        if df.loc[index]["Customer ID"] in trainID_list: 
            # if df.loc[index]["Customer ID"] == "2014-MKGMH": 
            #     print("find!")
            #     print(len(train))
            train.append(df.loc[index].values)
        else: test.append(df.loc[index].values)
    # train test sort 
    # print(len(train), len(train[0]))

    train, test = np.array(train), np.array(test)
    for i in range(len(train)):
        # print(train[:, 0].shape)
        idx = train[:, 0].tolist().index(trainID_list[i])
        train[[idx, i], :] = train[[i, idx], :]
    for i in range(len(test)):
        idx = test[:, 0].tolist().index(testID_list[i])
        test[[idx, i], :] = test[[i, idx], :]
    train = train[:, 1:]
    test = test[:, 1:]
    print(train.shape, test.shape)
    x = np.concatenate((x, train), axis=1)
    y = np.concatenate((y, test), axis=1)
    print(x, y)
    return x, y

train_feature, test_feature =  add_feature(train_feature, test_feature, services)
train_feature, test_feature = np.array(train_feature), np.array(test_feature)
print(train_feature.shape, test_feature.shape)
print(services.tail(5), services.head(5))

df_train = pd.DataFrame(data = train_feature)
df_test = pd.DataFrame(data = test_feature)

df_train.to_csv('train.csv')
df_test.to_csv('test.csv')

