import pandas as pd

demographics = pd.read_csv("data/demographics.csv")
location = pd.read_csv("data/location.csv")
population = pd.read_csv("data/population.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")
satisfaction = pd.read_csv("data/satisfaction.csv")
services = pd.read_csv("data/services.csv")
status = pd.read_csv("data/status.csv")
train_IDs = pd.read_csv("data/Train_IDs.csv")
test_IDs = pd.read_csv("data/Test_IDs.csv")


serviceDict = dict()
serviceAverage = [0 i in range(30)]
serviceLen = len(services)

for row in services.iterrows():
    for c in range(30):
        if isinstance(row[c], int) or isinstance(row[c], float):
            serviceAverage[c] += row[c]
for i in serviceAverage:
    i /= serviceLen + 500

for row in service.iterrows():
    if row[0] not in serviceDict.keys():
        serviceDict[row[0]] = []

    if row[4] == '':
        serviceDict[row[0]][0] = serviceAverage[4]
    else:
        serviceDict[row[0]][0] = row[4]

    if row[5] == '':
        serviceDict[row[0]][1] = serviceAverage[5]
    else:
        serviceDict[row[0]][1] = row[5]

    if row[8] == '':
        serviceDict[row[0]][2] = serviceAverage[8]
    else:
        serviceDict[row[0]][2] = row[8]

    if row[12] == '':
        serviceDict[row[0]][3] = serviceAverage[12]
    else:
        serviceDict[row[0]][3] = row[12]

    if row[24] == '':
        serviceDict[row[0]][4] = serviceAverage[24]
    else:
        serviceDict[row[0]][4] = row[24]

    if row[25] == '':
        serviceDict[row[0]][5] = serviceAverage[25]
    else:
        serviceDict[row[0]][5] = row[25]

    if row[26] == '':
        serviceDict[row[0]][6] = serviceAverage[26]
    else:
        serviceDict[row[0]][6] = row[26]

    if row[27] == '':
        serviceDict[row[0]][7] = serviceAverage[27]
    else:
        serviceDict[row[0]][7] = row[27]

    if row[28] == '':
        serviceDict[row[0]][8] = serviceAverage[28]
    else:
        serviceDict[row[0]][8] = row[28]

    if row[29] == '':
        serviceDict[row[0]][9] = serviceAverage[29]
    else:
        serviceDict[row[0]][9] = row[29]

statusType = []
for row in status.iterrows():
    if row[0] in serviceDict.keys():
        if row[1] not in statusType:
            statusType.append(row[1])
        serviceDict[row[0]][10] = statusType.index(row[1])

testDict = dict()
for id in test_IDs.iterrows():
    if id[0] in serviceDict.keys():
        testDict[id[0]] = serviceDict[id[0]]
        serviceDict.pop(id[0])

df_train = pd.DataFrame(data = serviceDict)
df_test = pd.DataFrame(data = testDict)

df_train.to_csv('extracted.csv')
df_test.to_csv('testData.csv')

# print(services)
# new_services = services.dropna()
# print(new_services)
