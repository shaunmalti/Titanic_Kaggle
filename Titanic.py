import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)

# Following examples set in Titanic Data Science Solutions seen here -> https://www.kaggle.com/startupsci/titanic-data-science-solutions#Titanic-Data-Science-Solutions


def main():
    test = pd.read_csv("./titanic/test.csv")
    train = pd.read_csv("./titanic/train.csv")


    # figures(train)
    # describe(train)
    replaceTitles(train)
    convertSex(train)
    predictAge(train)
    engineerFeature(train)
    bucketFareFeature(train)

def engineerFeature(data):
    # create family size feature and is alone feature
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


def bucketFareFeature(data):
    data['FareBand'] = pd.qcut(data['Fare'], 4)
    data.loc[(data['Fare']) <= 8, 'Fare'] = 0
    data.loc[(data['Fare'] > 8) &  ((data['Fare']) <= 14), 'Fare'] = 1
    data.loc[(data['Fare'] > 14) &  ((data['Fare']) <= 31), 'Fare'] = 2
    data.loc[(data['Fare'] > 31), 'Fare'] = 3
    print(data.head(10))

# to fill na ages either use median - as seen in https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
def predictAge(data):
    # from https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial, removing null values in embarked column
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # print(data.groupby(['Pclass', 'Sex']).median()['Age'])
    # age bands
    data['AgeBand'] = pd.cut(data['Age'], 5)
    # print(data.head(10))

    # filling in nulls in age col with median
    data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # print(data.isnull().sum())
    # print(data.loc[data['Embarked'].isna()])

    # changing age values to ordinals
    # (0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] <
    # (48.168, 64.084] < (64.084, 80.0]
    data.loc[(data['Age']) <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) &  ((data['Age']) <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 33) &  ((data['Age']) <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) &  ((data['Age']) <= 64), 'Age'] = 3
    data.loc[(data['Age'] > 64), 'Age'] = 4

def convertSex(data):
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # print(data.head())

# strongest correlation between parch and sibsp
# followed closely by fare and survived
# def figures(train):
    # describe(train)
    # correlationMatrix(train)
    # scatterPlots(train)

def describe(data):
    # print(data.columns.values)
    # print(data.info())
    print(data.describe())
    print('Length of train data')
    print(len(data))
    print('Length of train data age without Nas')
    print(len(data) - data['Age'].count())
    print(data.head(10))
    # difference in class shows survival rate increase - probably due to the fact that higher classes were placed
    # further up in the ship
    print(data[['Survived', 'Pclass']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    # doing the same but wrt to sex, women survived much more than men
    print(data[['Survived', 'Sex']].groupby(['Sex']).mean())
    # checking number of siblings in relation with survived metric
    print(data[['Survived', 'SibSp']].groupby(['SibSp']).mean())
    # checking number of kids in rel with survived metric
    print(data[['Survived', 'Parch']].groupby(['Parch']).mean())


def correlationMatrix(train):
    corrMatrix = train.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

def scatterPlots(data):
    # sns.scatterplot(x='Parch', y='SibSp', data=data)
    # plt.show()
    # sns.scatterplot(x='Fare', y='Survived', data=data)
    # plt.show()

    # show how age metric correlates with survived
    # g = sns.FacetGrid(data,col='Survived')
    # g.map(plt.hist, 'Age', bins=20)
    # plt.show()

    #
    # g = sns.FacetGrid(data, col='Survived', row='Pclass')
    # g.map(plt.hist, 'Age')
    # plt.show()

    # g = sns.FacetGrid(data, col='Survived')
    # g.map(plt.hist, 'Name')
    # plt.show()

    g = sns.FacetGrid(data, row= 'Pclass', col='Survived')
    g.map(plt.hist(plt.hist, 'Age'))


def replaceTitles(data):
    data['Name'] = data.Name.str.split('.').str.get(0).str.split(',').str.get(1).str.replace(' ', '').replace('\n', '')
    # data['Name'] = data['Name'].str.split('.').str.get(0).str.split(' ').str.get(1).str.replace(',','')
    # print(data.isnull().sum())
    # print(data.Name.head(10))
    data['Name'] = data['Name'].replace(['Master', 'Sir', 'Mr'], 0) # mr
    data['Name'] = data['Name'].replace(['Lady', 'theCountess', 'Mlle', 'Mme', 'Ms', 'Miss'], 1) # ms
    data['Name'] = data['Name'].replace(['Mrs'], 2) # mrs
    data['Name'] = data['Name'].replace(['Capt', 'Col',
                                         'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Don'], 3) # rare

    # print(data.Name.head(10))
    # print(data['Name'].unique())
    # print(data['Name'].unique())

if __name__ == '__main__':
    main()