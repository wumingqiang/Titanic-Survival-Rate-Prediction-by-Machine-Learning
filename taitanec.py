import sys
import sklearn
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("训练集大小：",train.shape,"测试集大小：",test.shape)
# 训练集大小： (891, 12) 测试集大小： (418, 11)

full = train._append(test,ignore_index = True)
print("合并后数据集大小",full.shape)
# 合并后数据集大小 (1309, 12)

# 输出前5行
#print(full.head())
#print(full.describe)
#print(full.info())

# 对于数据，使用平均值填充，对于分类，使用常见类填充
# Age年龄/Fare船票价格
full['Age'] = full['Age'].fillna(full['Age'].mean())
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
#print(full.info())


"""
    Embarked登船港口，共缺失2个，用类别最多的类填充(S)
    S(英国南安普顿)，C（法国瑟堡），Q(爱尔兰昆士敦)
    Cabin缺失比较多，填充为NAN
"""
#print(full['Embarked'].value_counts())
full['Embarked'] = full['Embarked'].fillna('S')
#print(full['Cabin'].value_counts())
full['Cabin'] = full['Cabin'].fillna('U')

# 全部属性填充完毕
#print(full.info())

"""
    特征提取：向量化
    数值类型：乘客编号（PassengerId）年龄（Age），船票价格（Fare），同代直系亲属人数（SibSp），不同代直系亲属人数（Parch）
    直接分类：乘客性别（Sex）：男性male(1)，女性female(0)
            登船港口（Embarked）：出发地点S=英国南安普顿Southampton，途径地点1：C=法国 瑟堡市Cherbourg，出发地点2：Q=爱尔兰 昆士敦Queenstown
            客舱等级（Pclass）：1=1等舱，2=2等舱，3=3等舱
    字符串类型：  乘客姓名（Name）
                客舱号（Cabin）
                船票编号（Ticket)
"""
sex_mapDict = {"male":1,"female":0}
full['Sex'] = full['Sex'].map(sex_mapDict)
# print(full['Sex'])

# 存放提取后的特征
embarkedDf = pd.DataFrame()
pclassDf = pd.DataFrame()
#print(embarkedDf.head())
# one-hot编码
embarkedDf = pd.get_dummies(full['Embarked'],prefix='Embarked')
pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass')
full = pd.concat([full,embarkedDf],axis = 1)
full = pd.concat([full,pclassDf],axis = 1)
full.drop('Embarked',axis = 1,inplace =True)
full.drop('Pclass',axis = 1,inplace =True  )
print(full.shape)
#print(full['Name'].head())

#定义函数，分割名字中的头衔
def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3

# 存放提取后的特征
titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
#print(titleDf.head())

'''
定义以下几种头衔类别：
Officer政府官员
Royalty王室（皇室）
Mr已婚男士
Mrs已婚妇女
Miss年轻未婚女子
Master有技能的人/教师
'''
#姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
# map映射：对Seris的每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
# 使用get_dummies进行one-hot 编码
titleDf = pd.get_dummies(titleDf['Title'])
full = pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace = True)
#print(full.head())
#print(titleDf.head())
print(full.shape)


#print(full['Cabin'].head())
cabinDf = pd.DataFrame()
full['Cabin'] = full['Cabin'].map(lambda c:c[0])
cabinDf = pd.get_dummies(full['Cabin'],prefix = 'Cabin')
full = pd.concat([full,cabinDf],axis = 1)
full.drop('Cabin',axis = 1,inplace = True)
#print(full.head())
#print(full.shape)


"""
    家庭人数 = 同代直系亲属（Parch） + 隔代亲属（SibSp） + 乘客自己
    孤家寡人 Family_Single :size == 1
    中等家庭 Family_Small :2<= size <=4
    大家庭 Family_Large :size > 4
"""
familyDf = pd.DataFrame()
familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s:1 if s>4 else 0)
full = pd.concat([full,familyDf],axis = 1 )
#print(full.head())
print(full.info())
print(full.shape)

# 删除 Ticket
full.drop('Ticket',axis = 1,inplace = True)


# 计算各个特征的相关系数
corrDf = full.corr()

# 将特征按照相关性排序
#print(corrDf['Survived'].sort_values(ascending = False))
"""
    选择如下特征作为主要特征
    头衔(TitleDf),客舱等级(PclassDf),、家庭大小(familyDf)
    船票价格(Fare),船舱号(cabinDf),登船港口(embarkedDf),
    性别(Sex)
    full_X是新的特征数据
"""

full_X = pd.concat([
        titleDf,
        pclassDf,
        familyDf,
        full['Fare'],
        cabinDf,
        embarkedDf,
        full['Sex'],],axis = 1)

# 保存特征工程处理后的数据
full_X.to_csv('data.csv', index=False)







