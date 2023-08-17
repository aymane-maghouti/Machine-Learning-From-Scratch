from LogisticRegression import  LogisticRegression,accuracy
from sklearn.model_selection import train_test_split
from sklearn import datasets

data = datasets.load_breast_cancer()
X ,y = data.data , data.target
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2,random_state=1234)

model = LogisticRegression(n_iters=10000,lr=0.1)
model.fit(X,y)
y_predict = model.predict(X_test)

acc = accuracy(y_predict,y_test)
print(acc)