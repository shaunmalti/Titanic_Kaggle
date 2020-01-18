import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

def main():
    test = pd.read_csv("./titanic/test.csv")
    train = pd.read_csv("./titanic/train.csv")
    figures(train)

# strongest correlation between parch and sibsp
# followed closely by fare and survived
def figures(train):
    describe(train)
    # correlationMatrix(train)
    # scatterPlots(train)


def describe(data):
    # print(data.columns.values)
    # print(data.info())
    print(data.describe())
    print(data[['Survived', 'Pclass']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))


def correlationMatrix(train):
    corrMatrix = train.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

def scatterPlots(data):
    # print(data.head())
    sns.scatterplot(x='Parch', y='SibSp', data=data)
    plt.show()
    sns.scatterplot(x='Fare', y='Survived', data=data)
    plt.show()


if __name__ == '__main__':
    main()