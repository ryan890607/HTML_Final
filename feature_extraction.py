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

# print(services)
# new_services = services.dropna()
# print(new_services)