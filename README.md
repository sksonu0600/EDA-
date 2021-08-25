# Exploratory Data Analysis



#load the libraries
import pandas as pd
import numpy as np
import pandas_profiling as pp
import sweetviz as sv

data1 = pd.read_csv("data_clean.csv")

data1

data1.tail(10)

data1

#Data Structure 
type(data1)
data1.shape

#data types
data1.dtypes

# Data type conversion

data1.info()

data1

data2=data1.iloc[:,1:]

data2

#The method .copy() is used here so that any changes made in new DataFrame don't get reflected in the original one
data=data2.copy()

data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data['Temp C']=pd.to_numeric(data['Temp C'],errors='coerce')# coerce will introduce NA values for non numeric data in the columns
data['Weather']=data['Weather'].astype('category')           #data['Wind']=data['Wind'].astype('int64')

data.info()

# Duplicates

#Count of duplicated rows
data[data.duplicated()].shape

data

#Print the duplicated rows
data[data.duplicated()]

data_cleaned1=data.drop_duplicates()

data_cleaned1.shape

# Drop columns

data_cleaned2=data_cleaned1.drop('Temp C',axis=1)

data_cleaned2

# Rename the columns

#rename the Solar column
data_cleaned3 = data_cleaned2.rename({'Solar.R': 'Solar'}, axis=1)

data_cleaned3

# Outlier Detection

# histogram of Ozone
data_cleaned3['Ozone'].hist()

#Box plot
data_cleaned3.boxplot(column=['Ozone'])

#Descriptive stat
data_cleaned3['Ozone'].describe()

data_cleaned3

#Bar plot
data['Weather'].value_counts().plot.bar()

# Missing Values and Imputation

import seaborn as sns
cols = data_cleaned3.columns 
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(data_cleaned3[cols].isnull(),
            cmap=sns.color_palette(colours))

data_cleaned3[data_cleaned3.isnull().any(axis=1)].head()

data_cleaned3.isnull().sum()

#Mean Imputation
mean = data_cleaned3['Ozone'].mean()
print(mean)

data_cleaned3['Ozone'] = data_cleaned3['Ozone'].fillna(mean)

data_cleaned3

#Missing value imputation for categorical vlaue
#Get the object columns
obj_columns=data_cleaned3[['Weather']]

obj_columns.isnull().sum()

#Missing value imputation for categorical vlaue
obj_columns=obj_columns.fillna(obj_columns.mode().iloc[0])

obj_columns.isnull().sum()

data_cleaned3.shape

obj_columns.shape

#Join the data set with imputed object dataset
data_cleaned4=pd.concat([data_cleaned3,obj_columns],axis=1)

data_cleaned4.isnull().sum()

# Scatter plot and Correlation analysis

# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(data_cleaned3)

#Correlation
data_cleaned3.corr()

# Transformations

#### Dummy Variable

#Creating dummy variable for Weather column
data_cleaned4=pd.get_dummies(data,columns=['Weather'])

data_cleaned4

data_cleaned4=data_cleaned4.dropna()

#### Normalization of the data

#Normalization of the data
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

data_cleaned4.values

array = data_cleaned3.values

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array[:,0:5])

#transformed data
set_printoptions(precision=2)
print(rescaledX[0:5,:])


# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler

array = data_cleaned4.values
scaler = StandardScaler().fit(array)
rescaledX = scaler.transform(array)

# summarize transformed data
set_printoptions(precision=2)
print(rescaledX[0:5,:])

# Speed up the EDA process

EDA_report= pp.ProfileReport(data)
EDA_report.to_file(output_file='report.html')

sweet_report = sv.analyze(data)
sweet_report.show_html('weather_report.html')
