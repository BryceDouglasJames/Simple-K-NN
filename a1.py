import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# import csv and create columns table
file = pd.read_csv("./iris.csv")
df = pd.DataFrame(file)
index = 0
col_vals = df.columns

# iterate columns and find the average of each element regardless of NaN found...
for col in df.columns:
    sum = 0.0
    count = 0
    for ele in df[col_vals[index]]:
        if(str(ele).replace('.', '', 1).isdigit()):
            count += 1
            sum += ele
    if count == 0:
        count = 1
    df[col_vals[index]] = df[col_vals[index]].fillna(round(sum/count, 1))
    index += 1


# label key fields
ans_map = ["Setosa", "Versicolor", "Virginica"]
df = df.replace("Setosa", 1)
df = df.replace("Versicolor", 2)
df = df.replace("Virginica", 3)

# create training and test sets
x = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
y = df['variety'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# generate K-NN model for training data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)


# REPL
print("Welcome to the iris.csv K-NN model! Please follow the instructions to \ninput data to predict upon... Enter '!DONE' to exit.")
print(df.head())
print("Accuracy: {}".format(neigh.score(x_test, y_test)))
sl = sw = pl = pw = 0.0
while(True):
    while(True):
        i = input("Enter value for Sepal length: ")
        if(str(i) == "!DONE"):
            exit()
        elif re.match(r'^-?\d+(?:\.\d+)$', i) is None:
            print("ERROR! Incompatible value...")
        else:
            sl = i
            break

    while(True):
        i = input("Enter value for Sepal width: ")
        if(i == "!DONE"):
            exit()
        elif re.match(r'^-?\d+(?:\.\d+)$', i) is None:
            print("ERROR! Incompatible value...")
        else:
            sw = i
            break

    while(True):
        i = input("Enter value for Petal length: ")
        if(i == "!DONE"):
            exit()
        elif re.match(r'^-?\d+(?:\.\d+)$', i) is None:
            print("ERROR! Incompatible value...")
        else:
            pl = i
            break

    while(True):
        i = input("Enter value for Petal width: ")
        if(i == "!DONE"):
            exit()
        elif re.match(r'^-?\d+(?:\.\d+)$', i) is None:
            print("ERROR! Incompatible value...")
        else:
            pw = i
            break

    ans = neigh.predict([[float(sl), float(sw), float(pl), float(pw)]])
#ans = neigh.predict([[5.1, 3.5, 1.4, 0.2]])
# print(ans[0])
    print("The model predicts that the data you entered is of variety {}".format(
        ans_map[ans[0]-1]))
