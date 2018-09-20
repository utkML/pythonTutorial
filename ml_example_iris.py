import pandas
from sklearn import model_selection
import sklearn.ensemble as models

train = pandas.read_csv("/home/manny/Desktop/iris.csv")

x = train.iloc[:,1:4]

y = train.iloc[:,-1]


y = pandas.factorize(y)[0]

xtrain, xtest, ytrain, ytest =  model_selection.train_test_split(x,y)


print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

model = models.RandomForestClassifier()
model.fit(xtrain,ytrain)

print(model.score(xtest,ytest))

