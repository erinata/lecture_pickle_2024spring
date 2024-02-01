import pandas
import pickle



with open("machine.pickle", "rb") as file:
  machine = pickle.load(file)


new_survey = pandas.read_csv("new_survey.csv")
new_survey = new_survey.values

print(machine.predict(new_survey))
