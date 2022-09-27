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

# Helper functions
response_map = ["Enter value for Sepal length: ",
                "Enter value for Sepal width: ",
                "Enter value for Petal length: ",
                "Enter value for Petal width: "]


def record_value(index: int) -> str:
    prompt = response_map[index]
    ans = ""
    while(True):
        i = input(prompt)
        if(str(i) == "!DONE"):
            exit()
        elif re.match(r'^-?\d+(?:\.\d+)$', i) is None:
            print("ERROR! Incompatible value...")
        else:
            ans = i
            break
    return ans


# REPL
print("\n\nWelcome to the iris.csv K-NN model! Please follow the instructions to input data to predict upon... Enter '!DONE' to exit.")
print(df.head())
print("Accuracy: {}".format(neigh.score(x_test, y_test)))
sl = sw = pl = pw = 0.0
while(True):
    sl = record_value(0)
    sw = record_value(1)
    pl = record_value(2)
    pw = record_value(3)
    ans = neigh.predict([[float(sl), float(sw), float(pl), float(pw)]])
    print("The model predicts that the data you entered is of variety {}".format(
        ans_map[ans[0]-1]))
