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
def add_feature(x, y, df, useless_col ,fillna=True, str_remain=True):
    # add data not in df but in train or test
    for id in tqdm(trainID_list):
        if id not in list(df["Customer ID"]): 
            tmp = [id]
            tmp.extend([np.nan for i in range(len(df.columns) - 1)])
            df.loc[len(df)] = tmp
    for id in tqdm(testID_list):
        if id not in list(df["Customer ID"]): 
            tmp = [id]
            tmp.extend([np.nan for i in range(len(df.columns) - 1)])
            df.loc[len(df)] = tmp

    # delete useless column
    df = df.drop(useless_col, axis=1)

    # for discrete continuous  col and useless column
    str_col = []
    # fill NaN
    if(fillna):
        for column_name in df.columns:
            if column_name == "Customer ID": continue
            for index in range(len(df)):
                if not df.isnull()[column_name][index]: break
            # discrete
            if isinstance(df.loc[index][column_name], str): 
                df[column_name] = df[column_name].fillna(df.mode().loc[0][column_name])
                str_col.append(column_name)
            # continuous
            else: 
                df[column_name] = df[column_name].fillna(df[column_name].mean())
                # df[column_name] = df[column_name].fillna(0)
    else:
        for column_name in df.columns:
            if column_name == "Customer ID": continue
            for index in range(len(df)):
                if not df.isnull()[column_name][index]: break
            # discrete
            if isinstance(df.loc[index][column_name], str): str_col.append(column_name)

    # one hot encoding
    if str_remain: df = pd.get_dummies(df, columns=str_col)
    else: df = df.drop(str_col, axis=1)

    # train test split
    train, test = [], []
    for index in tqdm(range(len(df))):
        if df.loc[index]["Customer ID"] in trainID_list: 
            # if df.loc[index]["Customer ID"] == "2014-MKGMH": 
            #     print("find!")
            #     print(len(train))
            train.append(df.loc[index].values)
        else: test.append(df.loc[index].values)

    # train test sort by TRAIN_IDs.csv and TEST_IDs.csv
    train, test = np.array(train), np.array(test)
    for i in range(len(train)):
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
    # print(x, y)
    return x, y

def add_label(x, df):
    # add data not in df but in train
    for id in tqdm(trainID_list):
        if id not in list(df["Customer ID"]): 
            df.loc[len(df)] = [id, np.nan]
    # print(df.info)


    # transform label into number
    label_idx_list = ["No Churn", "Competitor", "Dissatisfaction", "Attitude", "Price", "Other"]
    for index in tqdm(range(len(df))):
        # print(df.isnull()["Churn Category"][index])
        if not df.isnull()["Churn Category"][index]: df.loc[index]["Churn Category"] = label_idx_list.index(df.loc[index]["Churn Category"])

    # constuct label array
    label = []
    for index in tqdm(range(len(df))):
        label.append(df.loc[index].values)

    # train test sort by TRAIN_IDs.csv
    label = np.array(label)
    for i in range(len(label)):
        idx = label[:, 0].tolist().index(trainID_list[i])
        label[[idx, i], :] = label[[i, idx], :]
    label = label[:, 1:]
    print(label.shape)
    x = np.concatenate((x,  label), axis=1)
    return x

# add features
train_feature, test_feature =  add_feature(train_feature, test_feature, services, ["Referred a Friend", ], False, True)
train_feature, test_feature =  add_feature(train_feature, test_feature, satisfaction, [], False, True)
# # train_feature, test_feature =  add_feature(train_feature, test_feature, location, [])
train_feature, test_feature =  add_feature(train_feature, test_feature, demographics, [], False, True)

# add lebel to train
train_feature =  add_label(train_feature, status)

print(train_feature.shape, test_feature.shape)
# print(services.tail(5), services.head(5))

df_train = pd.DataFrame(data = train_feature)
df_test = pd.DataFrame(data = test_feature)

df_train.to_csv('./features/train.csv')
df_test.to_csv('./features/test.csv')

# tmp = pd.read_csv("./features/test.csv")
# datas = tmp.values
# print(datas[0], test_feature[0])
# print(tmp.info)

