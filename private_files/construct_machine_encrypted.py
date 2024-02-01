import pandas

from sklearn.ensemble import RandomForestClassifier

import pickle


dataset = pandas.read_csv("private_dataset.csv")

target = dataset.iloc[:,30].values
data = dataset.iloc[:,0:30].values


machine = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=11)
machine.fit(data, target)

with open("machine.pickle", "wb") as file:
  pickle.dump(machine, file)
  