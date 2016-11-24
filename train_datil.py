#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:01:34 2016

@author: Qu
"""

#PassengerId -- A numerical id assigned to each passenger.
#Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
#Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
#Name -- the name of the passenger.
#Sex -- The gender of the passenger -- male or female.
#Age -- The age of the passenger. Fractional.
#SibSp -- The number of siblings and spouses the passenger had on board.
#Parch -- The number of parents and children the passenger had on board.
#Ticket -- The ticket number of the passenger.
#Fare -- How much the passenger paid for the ticker.
#Cabin -- Which cabin the passenger was in.
#Embarked -- Where the passenger boarded the Titanic.
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

train = pd.read_csv('/Users/Qu/Downloads/train.csv')
test = pd.read_csv('/Users/Qu/Downloads/test.csv')

alpha =0.6
fig = plt.figure(figsize = (15,5))

train[train.Survived == 0].Age.value_counts().plot(kind = 'kde',color ='r',label ='died',alpha = alpha)
train[train.Survived == 1].Age.value_counts().plot(kind = 'kde',color ='b',label ='survived',alpha = alpha)
plt.xlabel('Age')
plt.ylabel('amount')
plt.title('what is the difference between died and survived?')
plt.legend(loc='best')

df_male_rate = train[train.Sex == 'male' ].Survived.value_counts(normalize = True).sort_index()
df_female_rate = train[train.Sex == 'female'].Survived.value_counts(normalize = True).sort_index()

df_male = train[train.Sex == 'male' ].Survived.value_counts().sort_index()
df_female = train[train.Sex == 'female'].Survived.value_counts().sort_index()

fig = plt.figure(figsize =(8,5))

df_female.plot(kind = 'barh',color = 'r',label = 'female',alpha = alpha)
df_male.plot(kind ='barh',color = 'b',label = 'male',alpha = alpha)
plt.ylabel('Sex')
#plt.set_xticklabels(['died','survived'])
plt.xlabel('amounts')
plt.title('what is the difference between male and female?')
plt.legend(loc ='best')

df_male =train[train.Sex == 'male']
df_female = train[train.Sex == 'female']

fig = plt.figure(figsize =(15,5))

ax1 = plt.subplot2grid((1,4),(0,0))
df_male[df_male.Pclass<3].Survived.value_counts().sort_index().plot(kind = 'bar',color = 'r',label= 'high-male',alpha = alpha)
ax1.set_xticklabels(['died','survived'])
ax1.set_ylim([0,350])
plt.legend(loc ='best')

ax2 = plt.subplot2grid((1,4),(0,1))
df_male[df_male.Pclass ==3].Survived.value_counts().sort_index().plot(kind = 'bar',color = 'b',label = 'low-male',alpha = alpha)
ax2.set_xticklabels(['died','survived'])
ax2.set_ylim([0,350])
plt.legend(loc ='best')

ax3 = plt.subplot2grid((1,4),(0,2))
df_female[df_female.Pclass <3].Survived.value_counts().sort_index().plot(kind = 'bar',color = 'g',label = 'high-female',alpha = alpha)
ax3.set_xticklabels(['died','survived'])
ax3.set_ylim([0,350])
plt.legend(loc ='best')

ax4 = plt.subplot2grid((1,4),(0,3))
df_female[df_female.Pclass ==3].Survived.value_counts().sort_index().plot(kind = 'bar',color = 'r',label = 'low-female',alpha = alpha)
ax4.set_xticklabels(['died','survived'])
ax4.set_ylim([0,350])
plt.legend(loc ='best')
plt.grid()
plt.tight_layout()


