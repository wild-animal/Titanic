# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:22:04 2016

@author: Qu
"""

import numpy as np 
import pandas as pd 
from pandas import DataFrame,Series
from sklearn import tree

train = pd.read_csv('/Users/Qu/Downloads/train.csv')
test = pd.read_csv('/Users/Qu/Downloads/test.csv')

#先观察数据
print(train.head())
print(train.describe())
print(test.head())
print(test.describe())

#存活率

number_survived = train.Survived.value_counts()
number_survived_rate = train.Survived.value_counts(normalize = True)

#按性别查看存活率

number_survived_male = train.Survived[train['Sex'] == 'male' ].value_counts()
number_survived_female = train.Survived[train['Sex'] == 'female'].value_counts()

#按年龄查看存活率
#新建一个年龄表

train['Child'] = float('nan')
train.Child[train.Age >= 18] = 0
train.Child[train.Age < 18 ] = 1

#转化数据类型，处理缺失值
#将性别，到达地点（Embarked）转化为数值类型

print(train.Sex.unique())
train.loc[train['Sex'] == 'male','Sex'] = 0
train.loc[train['Sex'] == 'female','Sex'] = 1

print(train['Embarked'].unique())
train['Embarked'] = train['Embarked'].fillna('S')

train.loc[train['Embarked'] == 'S','Embarked'] = 0
train.loc[train['Embarked'] == 'C','Embarked'] = 1
train.loc[train['Embarked'] == 'Q','Embarked'] = 2

print(train.describe())

#处理Age的缺失值

train.Age = train.Age.fillna(train['Age'].median())

#构建决策树
target = train.Survived.values

#改进1 增加家庭逃跑
train['Family'] = train.SibSp + train.Parch + 1
features_one = train[['Age','Sex','Fare','Embarked','Family']].values

my_tree_one = tree.DecisionTreeClassifier(criterion = 'entropy')
my_tree_one = my_tree_one.fit(features_one,target)

print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one,target))

#进行预测

test.Fare = test.Fare.fillna(test.Fare.median())

test.loc[test.Sex == 'male','Sex'] = 0
test.loc[test.Sex == 'female','Sex'] = 1
test.loc[test.Embarked == 'S','Embarked'] = 0
test.loc[test.Embarked == 'C','Embarked'] = 1
test.loc[test.Embarked == 'Q','Embarked'] = 2
test.Age = test.Age.fillna(test.Age.median())

#改进1 
test['Family'] = test.SibSp + test.Parch + 1 
test_features = test[['Age','Sex','Fare','Embarked','Family']].values

my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

test_PassagngerId = np.array(test.PassengerId).astype(int)

my_solution = pd.DataFrame(my_prediction,test_PassagngerId,columns = ['Survived'])

my_solution.to_csv("my_solution_two.csv", index_label = ["PassengerId"])

#生成决策树可视化
with open('Titanic','w') as f:
    f = tree.export_graphviz(my_tree_one,out_file = f)

