import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import f1_score

df1 = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\classification_model_1\Titanic-Dataset.csv")

df1["Age"] = df1["Age"].fillna(df1["Age"].median())
df1["Embarked"] = df1["Embarked"].fillna(df1["Embarked"].mode()[0])

x = df1.drop(["PassengerId" , "Name" , "Survived" , "Cabin" , "Ticket"] , axis=1)
y = df1["Survived"]

num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Sex', 'Embarked']

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size=0.2 , random_state=42)

preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(drop="first" , handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprocess" , preprocess) , ("logreg" , LogisticRegression(max_iter=10000))])

param_grid = {"logreg__C" : [0.001 , 0.01 , 0.1 , 1 ,10 ,100] , "logreg__penalty" : ["l2"] , "logreg__solver" : ["lbfgs"]}

grid = GridSearchCV(pipeline, param_grid , cv=5 , scoring="f1")

grid.fit(xtrain , ytrain)

model = grid.best_estimator_

ypred = model.predict(xtest)
print(ypred)


# ----------------------
# Evaluation Matric
# ----------------------
f1Score = f1_score(ytest , ypred)
print("F1 Score : ",f1Score)