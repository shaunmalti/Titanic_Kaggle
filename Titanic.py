import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# chain families together - that has an effect on survival rate
# check for different titles Mr. Mrs. Ms. etc therefore for title check from after , till .
file = "titanic/train.csv"

data = pd.read_csv(file, header=0)

data['Name'] = data.Name.str.split('.').str.get(0).str.split(',').str.get(1).replace(' ','').replace('\n','')

print(data.isnull().sum())

data['Name'] = data['Name'].replace(['Master','Sir'], 'Mr',regex=True)
data['Name'] = data['Name'].replace(['Lady','the Countess','Mlle','Mme','Ms'], 'Miss',regex=True)
data['Name'] = data['Name'].replace(['Capt', 'Col',
                                     'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Don'], 'Rare',regex=True)

null_age_dataframe = data[data['Age'].isnull()] # TODO null_age_dataframe contains rows where age is null

data = data.dropna(subset=['Age']) # TODO now data contains rows where age is not null


# first carry out linear regression to determine the age, missing ages are 177/891 = 19.86% of the training data
