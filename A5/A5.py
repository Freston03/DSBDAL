import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,confusion_matrix,precision_recall_fscore_support
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df= pd.read_csv('Social_Network_Ads.csv')
print(df)
#select features and target variables

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

print(X)
print(y)
# fit scaler on training data
norm = MinMaxScaler().fit(X)
# transform training data
X = norm.transform(X)

#splitting data into training and testing set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(y_train.shape)


#create a linear regression model

model=LogisticRegression()

#fit the model on training data
model.fit(X_train,y_train)

#make prediction on the test data

y_pred = model.predict(X_test)
print(y_pred)

# evaluate the model

mse = mean_squared_error(y_test,y_pred)

print(mse)

r_mse = np.sqrt(mse)

print(r_mse)

plt.scatter(y_test,y_pred,color='k')
plt.title('Logistic Regression Visualization')
plt.xlabel('actual values')
plt.ylabel('predicted values')

cf = confusion_matrix(y_test, y_pred)
print(cf)

score = precision_recall_fscore_support(y_test, y_pred, average='micro')
print('Precision of Model: ', score[0])
print('Recall of Model: ', score[1])
print('F-Score of Model: ', score[2])

sns.heatmap(cf,annot=True,fmt = 'd',cmap='Blues')
plt.xlabel("Acutual value")
plt.ylabel("predicted values")
plt.title("Confusion Matrics")
plt.show();

#

# Assignment number 5:
 # Load the Social network ads dataset
 # Display basic information
 # Display statistical information
 # Display null values
 # Fill the null values
 # Feature Engineering through correlation matrix 
 # Build the Logistic Regression Model and find its classification score
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
 xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=13) 
 from sklearn.linear_model import LogisticRegression
 model = LogisticRegression(solver = 'lbfgs')
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
df = pd.read_csv('purchase.csv')
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
df = df.drop('User ID', axis=1)
df.columns = ['Gender', 'Age', 'Salary', 'Purchased']
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Label encoding method
df['Gender']=df['Gender'].astype('category') 
df['Gender']=df['Gender'].cat.codes
# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
#---------------------------------------------------------------------------------------
# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
BuildModel(X, Y)
# #---------------------------------------------------------------------------------------
# Checking model score after removing outliers
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='Age', ax=axes[0])
sns.boxplot(data = df, x ='Salary', ax=axes[1])
fig.tight_layout()
plt.show()
df = RemoveOutlier(df, 'Age')
df = RemoveOutlier(df, 'Salary')
# You can use normalization method to improve the score
# salary -> high range
# age -> low range
# Normalization will smoothe both range salary and age
# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
BuildModel(X, Y)