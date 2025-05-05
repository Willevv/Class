import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import csv
import matplotlib.pyplot as plt
import seaborn as sb

"""Removing a random song-lyrics included in the file"""
with open('TrainOnMe-4.csv', "r") as f:
    with open('NewTrainFile.csv', "w") as w:
        lines = f.readlines()
        for i in range(len(lines)):
            if i >222 and i <= 230:
                pass
            else:
                w.write(lines[i])

"""Evaluation data"""
Xev = pd.read_csv("EvaluateOnME-4.csv",sep=",",usecols=["x1","x2","x3","x4","x5","x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13"])
Xev = Xev.drop(['x12'], axis = 1)
Xev = pd.get_dummies(Xev) # One-hot encoding of string values
Xev['x4']=pd.to_numeric(Xev['x4'], errors='coerce')

"""Training data"""
X = pd.read_csv("NewTrainFile.csv",sep=",",usecols=["x1", "x2", "x3","x4","x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13"])
X = X.drop(['x12'], axis = 1)
Y = pd.read_csv("NewTrainFile.csv", sep=",", usecols = ["y"])
X["x7"] = X["x7"].replace('olka', 'Polka')
X['x4'] = pd.to_numeric(X['x4'], errors='coerce') # replacing strin rep of floats to acual floats
X['x7'] = X['x7'].replace('olka', 'Polka')
X['x7'] = X['x7'].replace('chottis', 'Schottis')
X = pd.get_dummies(X) # One-hot encoding of string values

"""Drop Row with NaN, row 121 and 199 is removed"""
X = X.drop(121, axis = 0)
Y = Y.drop(121, axis = 0)
X  = X.drop(199, axis = 0)
Y = Y.drop(199, axis = 0)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, np.ravel(Y), test_size=0.3, random_state=0) #30% testing 70%training

""""GRadient boost classifier"""

Classifier = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=2,
            random_state=0).fit(X_train,y_train)

print(Classifier.classes_)

print(Classifier.score(X_test,y_test))

predicted = Classifier.predict(Xev)

print(Y.loc[0].at['y'])

with open('Label.txt', 'w') as f:
  for item in predicted:
    f.write("%s\n" % item)


import csv

with open('EvaluationGT-4.csv', 'r') as file:
    reader = csv.reader(file)
    data = [row[0] for row in reader]

# Now `data` is a list of rows (each row is a list of strings)
count = 0

for i in range(len(predicted)):
    if predicted[i] == data[i]:
        count = count+1

print(count)
print(f"Accuracy: {count/len(data):.4f}")


