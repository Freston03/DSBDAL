# Assignment number 2:
# Load the dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Quantization (Encoding): Convert categorical to numerical variable
# Handle outliers
# Handle skewed data 
#--------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#--------------------------------------------------------------------------------------
def DetectOutlier(df,var):    
    # IQR method is used to deal with outliers
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)
    # new dataframe is created which contains outliers
    df1 = df[((df[var] < low) | (df[var] > high))]  #these are outliers
    print('Outliers : \n', len(df1))
    print(df1.T)
    df = df[((df[var] >= low) & (df[var] <= high))]  #now filter out data which is not outlier
    return(df)
#--------------------------------------------------------------------------------------
df = pd.read_csv('academic.csv')
#--------------------------------------------------------------------------------------
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#--------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#--------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset: \n', df.isna().sum())
#--------------------------------------------------------------------------------------
# Fill the missing values
#df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df.fillna({'gender': df['gender'].mode()[0]}, inplace=True)
#df['raisedhands'].fillna(df['raisedhands'].mean(), inplace=True)
df.fillna({'raisedhands': df['raisedhands'].mean()}, inplace=True)
print('Total Number of Null Values in Dataset: \n', df.isna().sum())
#--------------------------------------------------------------------------------------
# Converting categorical to numeric using Find and replace method
df['Relation']=df['Relation'].astype('category')
df['Relation']=df['Relation'].cat.codes
#--------------------------------------------------------------------------------------
# Outliers can be visualized using boxplot
# using seaborn library we can plot the boxplot
fig, axes = plt.subplots(2,2)
fig.suptitle('Before removing Outliers')
sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
sns.boxplot(data = df, x ='VisITedResources', ax=axes[0,1])
sns.boxplot(data = df, x ='AnnouncementsView', ax=axes[1,0])
sns.boxplot(data = df, x ='Discussion', ax=axes[1,1])
plt.show()
#Display and remove outliers
df = DetectOutlier(df, 'raisedhands')
fig, axes = plt.subplots(2,2)
fig.suptitle('After removing Outliers')
sns.boxplot(data = df, x ='raisedhands', ax=axes[0,0])
sns.boxplot(data = df, x ='VisITedResources', ax=axes[0,1])
sns.boxplot(data = df, x ='AnnouncementsView', ax=axes[1,0])
sns.boxplot(data = df, x ='Discussion', ax=axes[1,1])
plt.show()
#--------------------------------------------------------------------------------------
print('---------------- Data Skew Values before Yeo John Transformation ----------------------')
# There are two types
# 1. Left skew
# 2. Right skew
# Formula to find out data skewness = 3*(mean-median)/std                           
     # = 0(no skew) print        
     # = negative (Negative skew)  left skewed data                           
     # = positve (Positive skew)   Right skewed data
     # = -0.5 to 0 to 0.5  (acceptable skew)
     # = -0.5> <-1 moderate negative skew 
     # = 0.5> <1 moderate positive skew
     # = > -1 high negative
     # = > 1 high positive

print('raisedhands: ', df['raisedhands'].skew())
print('VisITedResources: ', df['VisITedResources'].skew())
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())
fig, axes = plt.subplots(2,2)
fig.suptitle('Handling Data Skewness')
sns.histplot(ax = axes[0,0], data = df['AnnouncementsView'], kde=True)
sns.histplot(ax = axes[0,1], data = df['Discussion'], kde=True)
from sklearn.preprocessing import PowerTransformer
yeojohnTr = PowerTransformer(standardize=True)  
df['AnnouncementsView'] = yeojohnTr.fit_transform(df['AnnouncementsView'].values.reshape(-1,1))
df['Discussion'] = yeojohnTr.fit_transform(df['Discussion'].values.reshape(-1,1))
print('---------------- Data Skew Values after Yeo John Transformation ----------------------')
print('AnnouncementsView: ', df['AnnouncementsView'].skew())
print('Discussion: ', df['Discussion'].skew())
sns.histplot(ax = axes[1,0], data = df['AnnouncementsView'], kde=True)
sns.histplot(ax = axes[1,1], data = df['Discussion'], kde=True)
plt.show()
