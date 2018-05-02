import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# chain families together - that has an effect on survival rate
# check for different titles Mr. Mrs. Ms. etc therefore for title check from after , till .
def main():
    file = "titanic/train.csv"
    file_test = "titanic/test.csv"
    data = pd.read_csv(file, header=0)


    data['Name'] = data.Name.str.split('.').str.get(0).str.split(',').str.get(1).replace(' ','').replace('\n','')

    data['Name'] = data['Name'].replace(['Master','Sir','Dr', 'Major','Col'], 'Mr',regex=True)
    data['Name'] = data['Name'].replace(['Lady','the Countess','Mlle','Mme','Ms'], 'Miss',regex=True)
    data['Name'] = data['Name'].replace(['Capt',
                                         'Don', 'Rev', 'Jonkheer'], 'Rare',regex=True)

    null_age_dataframe = data[data['Age'].isnull()] # TODO null_age_dataframe contains rows where age is null

    data = data.dropna(subset=['Age']) # TODO now data contains rows where age is not null


    #mapping the name
    title_mapping = {" Mr": 1, " Miss": 2, " Mrs": 3, " Rare": 4}
    # data['Name'] = data['Name'].map(title_mapping)
    data['Name'] = data['Name'].astype(str).replace(' ','')
    data['NewNameCol'] = pd.Series(data['Name'].map(title_mapping),index=data.index)


    #
    y_train = data['Age']
    x_train = data[['Survived','Pclass','Name','NewNameCol']]
    # ols = linear_model.LinearRegression()
    # model = ols.fit(x_train,y_train)

    # things that effect the age - if survived, what class, the title (man or woman)

    # first carry out linear regression to determine the age, missing ages are 177/891 = 19.86% of the training data

if __name__ == '__main__':
    main()