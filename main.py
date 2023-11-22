# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv('/kaggle/input/telcom-churns-dataset/TelcoChurn.csv')
df.head()

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df.shape

df.info()

df = df.drop(['customerID'], axis = 1)
df.head()

df.isnull().sum()

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()

df[np.isnan(df['TotalCharges'])]

df.fillna(df["TotalCharges"].mean(),inplace=True)
#df.dropna(inplace = True)

plt.figure(figsize=(25, 10))
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, 
                 linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)

df.columns

x=['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
le = LabelEncoder()
for i in x:
    df[i] = le.fit_transform(df[i])
df.head()

y = df['Churn'].values
X = df.drop(columns = ['Churn'])

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

models = {
    'logistic regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
    'ANN' : MLPClassifier(),
    'KNN' : KNeighborsClassifier(),
    'naive bayes' : GaussianNB(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'{name} trained')
    prediction_test = model.predict(X_test)
    print ('Accuracy:',accuracy_score(y_test, prediction_test))
    print('Classification report:',classification_report(y_test, prediction_test))

from sklearn.neural_network import MLPClassifier
clf5 = MLPClassifier()
clf5.fit(X_train, y_train)
accuracy = clf5.score(X_test, y_test)
print(accuracy) 

df = df.drop(['Partner'], axis = 1)
df

df.info()

