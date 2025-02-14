# Assignment number 6:
# Load the Iris dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through correlation matrix
# Build the Gaussian Naive Bayes Model and find its classification score
# Remove outliers and again see the accuracy of the model
#---------------------------------------------------------------------------------------
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#-------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('iris.csv')

# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)

print("------------------Defining X and Y variable------------------")
X= df.drop(['Species'],axis=1)
Y= df.drop(['Id','SepalLength','SepalWidth','PetalLength','PetalWidth'],axis=1)
print("X:\n",X)
print("Y:\n",Y)
print("X.shape:",X.shape)
print("Y.shape:",Y.shape)
scaler = StandardScaler()
x = scaler.fit_transform(X.values)

print("---------------Split Train and Test Dataset---------------")
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,shuffle=True)
print("X Train Shape:",X_Train.shape)
print("X Test Shape:",X_Test.shape)
print("Y Train Shape:",Y_Train.shape)
print("Y Test Shape:",Y_Test.shape)

print("-----------------Apply Gaussian Naive Bayes----------------------------")
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_Train, Y_Train)

from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
Y_Pred=model.predict(X_Test)
A=model.score(X_Test,Y_Test)
print(A)
print(Y_Pred)
# print("Accuracy:",accuracy_score(Y_Test,Y_Pred))
# cm=confusion_matrix(Y_Test,Y_Pred)
# disp=ConfusionMatrixDisplay(confusion_matrix=cm)
# print("---------------------------COnfusion Matrix--------------")
# print(cm)
# disp.plot()
# plt.show()

# def get_confusion_matrix_values(Y_True,Y_Pred):
#     cm=confusion_matrix(Y_True,Y_Pred)
#     return(cm[0][0],cm[0][1],cm[1][0],cm[1][1])

# print("---------------Calculate TP,FP,FN,TN,Accuracy,Precision,Recall----------------------")
# TP,FP,FN,TN=get_confusion_matrix_values(Y_Test,Y_Pred)
# P=TP/(TP+FP)
# R=TP/(TP+FN)
# print("TP:",TP)
# print("FP:",FP)
# print("FN:",FN)
# print("TN:",TN)
# print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
# print("Precision:",TP/(TP+FP))
# print("Recall:",TP/(TP+FN))
# print("F1 Score:",(2*P*R)/(P+R))






# Assignment number 6:
 # Load the Iris dataset
 # Display basic information
 # Display statistical information
 # Display null values
 # Fill the null values
 # Feature Engineering through correlation matrix 
 # Build the Gaussian Naive Bayes Model and find its classification score
 # Remove outliers and again see the accuracy of the model
#---------------------------------------------------------------------------------------
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
def RemoveOutlier(df,var):
 Q1 = df[var].quantile(0.25)
 Q3 = df[var].quantile(0.75)
 IQR = Q3 - Q1
 high, low = Q3+1.5*IQR, Q1-1.5*IQR
 
 print("Highest allowed in variable:", var, high)
 print("lowest allowed in variable:", var, low)
 count = df[(df[var] > high) | (df[var] < low)][var].count()
 print('Total outliers in:',var,':',count)
 df = df[((df[var] >= low) & (df[var] <= high))]
 return df
#---------------------------------------------------------------------------------------
def BuildModel(X, Y):
 # Training and testing data
 from sklearn.model_selection import train_test_split
 # Assign test data size 20%
 xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=0) 
 # from sklearn.linear_model import LogisticRegression
 # model = LogisticRegression(solver = 'lbfgs')
 from sklearn.naive_bayes import GaussianNB
 model = GaussianNB()
 model = model.fit(xtrain,ytrain)
 ypred = model.predict(xtest)
 
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(ytest, ypred)
 sns.heatmap(cm, annot=True)
 plt.show()
 from sklearn.metrics import classification_report
 print(classification_report(ytest, ypred))
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Label encoding method
df['Species']=df['Species'].astype('category') 
df['Species']=df['Species'].cat.codes
# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
#---------------------------------------------------------------------------------------
# Choosing input and output variables from correlation matrix
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
#---------------------------------------------------------------------------------------
# Checking model score after removing outliers
fig, axes = plt.subplots(2,2)
sns.boxplot(data = df, x ='SL', ax=axes[0,0])
sns.boxplot(data = df, x ='SW', ax=axes[0,1])
sns.boxplot(data = df, x ='PL', ax=axes[1,0])
sns.boxplot(data = df, x ='PW', ax=axes[1,1])
plt.show()
df = RemoveOutlier(df, 'SW')
# Choosing input and output variables from correlation matrix
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
#After removing outliers accuracy is reducing due to overfitting of the model
