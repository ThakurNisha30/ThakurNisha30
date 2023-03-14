### Hi there 👋

<!--
**ThakurNisha30/ThakurNisha30** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:
# Data Preprocessing

#we have imported the house rent dataset csv file using pandas and read the file. Header is considered to be 0 as the first coumns are the heading of the data files
import pandas as pd

df = pd.read_csv('House_Rent_Dataset final.csv', header=0)

#head is used to see the first 5 rows of the dataset
df.head()

# Data frame . shape helps us to understand the number of rows and columns in the data set
print('The number of rows and columns respectively are ', df.shape)

#The data tells about the information regarding the dataset i.e The datatype of the variables, The number of rows and coulmn, The null values details
print('Data frame information: ')

df.info()

print("Data Frame Description"
      " , BHK means 1 Bathroom, Hall and Kitchen", 
      ', Size is in form of Square feet', 
      'Bathroom is the number of washrooms in the house')

df.describe()

The describe function tells about the description of the dataset i.e 
- The dataset which included the variables have 4746 rows
- the mean of BHK is 2.08
- the standard deviation is 0.8 for BHK
- BHK(Bedroom, Hall, Kitchen) where the min BHK is 1 and max BHK is 6
-  25%
-  50%
-  75%


#isnull command tells about the null values presecence in the dataset, which returns as boolean values, if there is a null valuee then the output will be True. But it is hard to understand from this information
df.isnull()


#The sum of the null values in the dataset can be understood through the below commend, where there are no null vales in the dataset
df.isnull().sum()

#Checking the duplicated values in the dataset is very important, we found that there are no duplicated values in the dataset
df.duplicated()

df.duplicated().sum()

import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

sns.boxplot(df['Rent'])

Box Plot is the visual representation of the depicting groups of numerical data through their quartiles. Boxplot is also used for detect the outlier in data set. It captures the summary of the data efficiently with a simple box and whiskers and allows us to compare easily across groups. Boxplot summarizes a sample data using 25th, 50th and 75th percentiles. These percentiles are also known as the lower quartile, median and upper quartile.

A box plot consist of 5 things.

- Minimum
- First Quartile or 25%
- Median (Second Quartile) or 50%
- Third Quartile or 75%
- Maximum

Analysis of Box plot : 
- The dataset vaiable rent has outliers as per the box plot (indicated by diamond dots, which has to be eliminated for accurary of the data)

df.drop(df[(df['Rent'] > df['Rent'].mean() + 2 * df['Rent'].std()) | (df['Rent'] < df['Rent'].mean() - 2 * df['Rent'].std())].index, inplace=True)

df.shape

sns.boxplot(df['Rent'])

Let’s take the first box plot i.e, blue box plot of the figure and understand these statistical things:

- Bottom black vertical line of blue box plot is minimum value
- First black vertical line of rectangle shape of blue box plot is First quartile or 25%
- Second black vertical line of rectangle shape of blue box plot is Second quartile or 50% or median.
- Third black vertical line of rectangle shape of blue box plot is third quartile or 75%
- Top black vertical line of rectangle shape of blue box plot is maximum value.
- Small diamond shape of blue box plot is outlier data or erroneous data.

# The +2SD and -2SD data is removed 
df.drop(df[(df['Rent'] > df['Rent'].mean() + 2 * df['Rent'].std()) | (df['Rent'] < df['Rent'].mean() - 2 * df['Rent'].std())].index, inplace=True)

df.shape

sns.boxplot(df['Rent'])

df.drop(df[(df['Rent'] > df['Rent'].mean() + 2 * df['Rent'].std()) | (df['Rent'] < df['Rent'].mean() - 2 * df['Rent'].std())].index, inplace=True)

df.shape

sns.boxplot(df['Rent'])

df.drop(df[(df['Rent'] > df['Rent'].mean() + 2 * df['Rent'].std()) | (df['Rent'] < df['Rent'].mean() - 2 * df['Rent'].std())].index, inplace=True)

df.shape

sns.boxplot(df['Rent'])

sns.boxplot(df['Size'])

df.drop(df[(df['Size'] > df['Size'].mean() + 2 * df['Size'].std()) | (df['Size'] < df['Size'].mean() - 2 * df['Size'].std())].index, inplace=True)

df.shape

sns.boxplot(df['Size'])

df.shape

df.info()

df.duplicated().sum()

df.dtypes

df.head()

df = df.replace({
                'Bachelors/Family':0,'Bachelors':1,'Family':2,
                'Super Area':0, 'Carpet Area':1, 'Built Area':2,   
                'Contact Owner':0,'Contact Agent':1,'Contact Builder':2,
                'Unfurnished':0,'Semi-Furnished':1,'Furnished':2,
                'Kolkata':0,'Mumbai':1,'Bangalore':2,'Delhi':3,'Chennai':4,'Hyderabad':5
            })

# Tried to do a dictionary connecting the keys and values for an understanding to know what variables is assigned with digital number
areaDict = {'0': "Super Area",'1': "Carpet Area",'2': "Built Area"}
TenantPreferredDict = {'0': "Bachelors/Family",'1': "Bachelors",'2': "Family"}
PointofContactDict = {'0': "Contact Owner",'1': "Contact Agent",'2': "Contact Builder"}
FurnishingStatusDict = {'0': "Unfurnished",'1': "Semi-Furnished",'2': "Furnished"}
CityDict = {'0': "Kolkata",'1': "Mumbai",'2': "Bangalore",'3': "Delhi",'4': "Chennai",'5': "Hyderabad"}

df.head()

#Splitting posted on column and creating new variables

df['Floor_Level']=df['Floor'].apply(lambda x: x.split('out of')[0])
df['Total_Floor']=df['Floor'].apply(lambda x: x.split('out of')[-1])
df.head()

df.drop(['Floor','Posted On', ],axis=1,inplace= True)
df.head()

df['Floor_Level']=df['Floor_Level'].apply(lambda x: 0 if x =='Ground'or x=='Ground 'or x=='Upper Basement ' or x=='Lower Basement ' else x)                          
df.head()

df['Total_Floor']=df['Total_Floor'].apply(lambda x: 0 if x =='Ground'or x=='Ground 'or x=='Upper Basement ' or x=='Lower Basement ' else x)                          
df.head()

df['Area Locality'].unique().shape

df.drop(columns = ['Area Locality'],inplace=True)

df.head()

df.shape

df.dtypes

df['Floor_Level'] = df['Floor_Level'].astype(int)
df['Total_Floor'] = df['Total_Floor'].astype(int)

df.dtypes

df.describe()

df. rename(columns = {'Area Type':'Area_Type', 'Point of Contact':'Point_of_Contact', 'Furnishing Status':'Furnishing_Status','Tenant Preferred' : 'Tenant_Preferred'}, inplace = True)

df

df.to_csv('House_Rent_Dataset logistic.csv',encoding='utf-8')

# Data Visualisation

#It means that there are no null values in the map
import seaborn as sns
sns.heatmap(df.isnull())

# Tried to do a dictionary connecting the keys and values for an understanding to know what variables is assigned with digital number
areaDict = {'0': "Super Area",'1': "Carpet Area",'2': "Built Area"}
TenantPreferredDict = {'0': "Bachelors/Family",'1': "Bachelors",'2': "Family"}
PointofContactDict = {'0': "Contact Owner",'1': "Contact Agent",'2': "Contact Builder"}
FurnishingStatusDict = {'0': "Unfurnished",'1': "Semi-Furnished",'2': "Furnished"}
CityDict = {'0': "Kolkata",'1': "Mumbai",'2': "Bangalore",'3': "Delhi",'4': "Chennai",'5': "Hyderabad"}

# to understand what % of super area, Buildarea and carpet area are present in area type , and so on using pie charts

col = ['Area_Type','City','Furnishing_Status','Tenant_Preferred','Point_of_Contact','BHK','Bathroom', 'Floor_Level', 'Total_Floor']
plt.figure(figsize=(30,30))
for i, col in enumerate(col):
    axes = plt.subplot(5,2,i + 1)
    axes.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%.2f%%')
    axes.set_title(col)
plt.tight_layout()
plt.legend()
plt.show()

#Histogram to know what is the contribution of each factor in area type.

col = ['Area_Type','City','Furnishing_Status',
           'Tenant_Preferred','Point_of_Contact','Bathroom','BHK','Floor_Level', 'Total_Floor']

plt.figure(figsize=(25,25))
for i, col in enumerate(col):
    axes = plt.subplot(6,2, i + 1)
    sns.countplot(x=df[col],ax=axes)
plt.tight_layout()
plt.show()

#A pairplot plot a pairwise relationships in a dataset. The pairplot function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column.

#df = pd.read_csv('House_Rent_Dataset visualisation.csv', header=0)

sns.pairplot(df, x_vars = ['Area_Type','City','Furnishing_Status',
           'Tenant_Preferred','Point_of_Contact','Bathroom','BHK','Floor_Level', 'Total_Floor'], y_vars = 'Rent',size = 5, aspect =0.7,kind="reg")

plt.hist(df['Rent'], bins=20)
plt.xlabel('Rent')

# Statistical Testing

import seaborn as sns

sns.distplot(df['Rent'], label='Rent')

plt.legend(prop={'size':12})
plt.title('Distribution of Rent')
plt.xlabel('Rent')
plt.ylabel('Density')

import seaborn as sns

sns.distplot(df['Size'], label='Size')

plt.legend(prop={'size':12})
plt.title('Distribution of Size')
plt.xlabel('Size')
plt.ylabel('Density')

Null Hypothesis: When looking at rental properties in India, the factors we examined do not demonstrate an association between the property's price and specific amenities; those factors cannot be used to predict the overall cost of the property.
  
  Alternate Hypothesis: When looking at rental properties in India, the factors we examined demonstrate an association between the property's price and specific amenities; those factors can be used to predict the overall cost of the property.

fvalue, pvalue = stats.f_oneway(df['Rent'], df['BHK'], df['Size'], df['Bathroom'], df['Area_Type'], df['Furnishing_Status'], df['Tenant_Preferred'], df['Point_of_Contact'], df['City'])
print('F-Value: ', fvalue, 'P-Value: ', pvalue)

#Hypothesis Test
from scipy import stats
k2, p = stats.f_oneway(df['Rent'], df['BHK'], df['Size'], df['Bathroom'], df['Area_Type'], df['Furnishing_Status'], df['Tenant_Preferred'], df['Point_of_Contact'], df['City'])
print('F-Value: ', fvalue, 'P-Value: ', pvalue)
alpha = 0.05
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Annova Testing

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Rent ~ C(BHK) + C(Size) + C(Bathroom) + C(City) + C(Point_of_Contact) + C(Furnishing_Status) + C(Tenant_Preferred) + C(Area_Type)', data=df).fit()
sm.stats.anova_lm(model, typ=2)

!pip install scipy

from scipy import stats
import scipy.stats as stats
from scipy.stats import f_oneway

fvalue, pvalue = stats.f_oneway(df['Rent'], df['BHK'], df['Size'], df['Bathroom'], df['Area_Type'], df['Furnishing_Status'], df['Tenant_Preferred'], df['Point_of_Contact'], df['City'])
print('F-Value: ', fvalue, 'P-Value: ', pvalue)

# Correlation

#To understand what variables are closer to the rent, a correlation is done
#corr_matrix
df = pd.read_csv('House_Rent_Dataset correlation.csv', header=0)

corr_matrix = df.corr()
corr_matrix['Rent'].sort_values(ascending=False)

df.corr()

ax = sns.heatmap(corr_matrix,linewidth=0.10)
plt.show()

df = df.drop(['Area_Type','Furnishing_Status','Tenant_Preferred','City'], axis=1)

df.head()

# Multiple Linear Regression Model Development
As the Bathroom, Total floor, Point of Contact, BHK, floor size and Size are correlated to the rent. We are considering the variables in the model development and dropping the rest


import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import ttest_1samp

fig = sm.qqplot(df, line='45')
plt.show()

df.head()

#Output
y = df['Rent']
df = df.drop(['Rent'], axis=1)

#This means the Size and BHK are considering by dropping the Size and BHK
df1 = df.drop(['Point_of_Contact','Floor_Level', 'Total_Floor'], axis = 1)
df2 = df.drop(['Size','Point_of_Contact'], axis = 1)
df3 = df.drop(['Size','Bathroom'], axis = 1)
df4 = df.drop(['BHK','Point_of_Contact', 'Floor_Level'], axis = 1)
df5 = df.drop(['BHK','Bathroom'], axis = 1)
df6 = df.drop(['BHK','Size'], axis = 1)
#Considering all factors
df7 = df

#this data keeps 20% data in testing and 80% in training

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sklearn
from sklearn import linear_model

X_train1, X_test1, y_train1, y_test1 = train_test_split(df1, y, test_size=0.2, random_state=21)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2, y, test_size=0.2, random_state=21)
X_train3, X_test3, y_train3, y_test3 = train_test_split(df3, y, test_size=0.2, random_state=21)
X_train4, X_test4, y_train4, y_test4 = train_test_split(df4, y, test_size=0.2, random_state=21)
X_train5, X_test5, y_train5, y_test5 = train_test_split(df5, y, test_size=0.2, random_state=21)
X_train6, X_test6, y_train6, y_test6 = train_test_split(df6, y, test_size=0.2, random_state=21)
X_train7, X_test7, y_train7, y_test7 = train_test_split(df7, y, test_size=0.2, random_state=21)

#This is formula, i am fitting the data 
model1 = linear_model.LinearRegression()
model1.fit(X_train1, y_train1)

model2 = linear_model.LinearRegression()
model2.fit(X_train2, y_train2)

model3 = linear_model.LinearRegression()
model3.fit(X_train3, y_train3)

model4 = linear_model.LinearRegression()
model4.fit(X_train4, y_train4)

model5 = linear_model.LinearRegression()
model5.fit(X_train5, y_train5)

model6 = linear_model.LinearRegression()
model6.fit(X_train6, y_train6)

model7 = linear_model.LinearRegression()
model7.fit(X_train7, y_train7)

y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)
y_pred3 = model3.predict(X_test3)
y_pred4 = model4.predict(X_test4)
y_pred5 = model5.predict(X_test5)
y_pred6 = model6.predict(X_test6)
y_pred7 = model7.predict(X_test7)

#R2 score tells how good the data has fit the regression line
#R2 score if near to 1, the model is better

r1 = r2_score(y_test1, y_pred1)
print('The R2 of model1 is :',r1)

r2 = r2_score(y_test2, y_pred2)
print('The R2 of model2 is :',r2)

r3 = r2_score(y_test3, y_pred3)
print('The R2 of model3 is :',r3)

r4 = r2_score(y_test4, y_pred4)
print('The R2 of model4 is :',r4)

r5 = r2_score(y_test5, y_pred5)
print('The R2 of model5 is :',r5)

r6 = r2_score(y_test6, y_pred6)
print('The R2 of model6 is :',r6)

r7 = r2_score(y_test7, y_pred7)
print('The R2 of model7 is :',r7)

# Prediction
As the model df7 performed well with 46.9% accurary compared to the other model, we will be predicting the rent with the model df7 which included independent variables i.e BHK, Size , Bathroom and Point of contact. 
- fOR 2 bhk, 1100Squarefeet size, 2 Bathroom and 0 Point of contact means contact owner,1st floor at 2 floored building - The rent will be 15031 per month

df_temp = pd.DataFrame([[2,1100,2,0,1,2]])

model7.predict(df_temp)

df_temp = pd.DataFrame([[1,1100,2,0,0,2]])
model7.predict(df_temp)

As the Co-efficient of Determination i.e R2 of model7 (Consideration of all independent variables) is 46.9%. so we can just that it is not a good model to predict the rent.
- As the variables are not closely correlated, it can also be reason where we couldnt predict accurate rent with model
- The dataset contains very less data i.e 4647 rows and we require more data to predict

# Logistic Regression

As we couldnt get good accurary from the multiple linear regression, we tried doing logistic regression

import pandas as pd

df = pd.read_csv('House_Rent_Dataset logistic.csv', header=0)

df.info()

df.describe()

df.loc[df['Rent'].between(1500,6500), 'Rent_group'] = '1'
df.loc[df['Rent'].between(6500,11500), 'Rent_group'] = '2'
df.loc[df['Rent'].between(11500,16500), 'Rent_group'] = '3'
df.loc[df['Rent'].between(16500,21500), 'Rent_group'] = '4'
df.loc[df['Rent'].between(21500,26500), 'Rent_group'] = '5'
df.loc[df['Rent'].between(26500,31500), 'Rent_group'] = '6'
df.loc[df['Rent'].between(31500,36500), 'Rent_group'] = '7'
df.loc[df['Rent'].between(36500,41500), 'Rent_group'] = '8'
df.head()

df = df[['BHK','Rent','Rent_group','Size','Area_Type', 'City','Furnishing_Status','Tenant_Preferred','Bathroom', 'Point_of_Contact','Floor_Level','Total_Floor']]

import pandas as pd

df.drop(['Rent'],axis=1,inplace= True)

df.head()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

X = df[['BHK', 'Size', 'Bathroom', 'Area_Type', 'Furnishing_Status', 'Tenant_Preferred', 'Point_of_Contact', 'City']]
y = df['Rent_group']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0) 

log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)
pred = log_regression.predict(X_test)
print(pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

#test prediction 
pred_single = log_regression.predict([[2,800,2,1,1,0,0,1]])
print(pred_single)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

r7 = r2_score(y_test, pred)
print('The R2 of model7 is :',r7)

#chi-square test for hypothesis testing
import pandas as pd 
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['Rent_group'], df['BHK'])
c, p, dof, expected = chi2_contingency(contingency_table)
print('chi square statistic: ', c)
print('pvalue: ' ,p)
print('degree of freedom: ', dof)
print('expected values: ', expected)

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# calculate roc curves
fpr, tpr, thresholds = roc_curve(df['Rent_group'], df['Area_Type'])
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

from scipy import stats
import scipy.stats as stats
from scipy.stats import f_oneway

fvalue, pvalue = stats.f_oneway(df['Rent_group'], df['BHK'], df['Size'], df['Bathroom'], df['Area_Type'], df['Furnishing_Status'], df['Tenant_Preferred'], df['Point_of_Contact'], df['City'])
print('F-Value: ', fvalue, 'P-Value: ', pvalue)




