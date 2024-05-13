import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def getData():
    sourceRow = 891
    source_X = pd.read_csv('data.csv')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    full = train._append(test,ignore_index = True)

    # 训练集有特征有标签
    all_x = source_X.loc[0:sourceRow-1,:]
    all_y = full.loc[0:sourceRow - 1,'Survived']

    #预测集只有特征
    pred_x = source_X.loc[sourceRow:,:]
    print("原始数据集行数：",all_x.shape[0])
    print("预测数据集行数：",pred_x.shape[0])
    return all_x,all_y



def LogReg(all_x,all_y,max_iter = 3000,):
    train_X, test_X, train_y, test_y = train_test_split(all_x,
                                                        all_y,
                                                        train_size=0.8)
    # 机器学习算法：逻辑回归
    model = LogisticRegression(max_iter=3000)
    # 训练模型
    model.fit(train_X, train_y)
    # 评估算法
    accuracy = model.score(test_X, test_y)
    print("Logistic Aucc:", accuracy)

def KNN(all_x,all_y,k = 5):
    train_X, test_X, train_y, test_y = train_test_split(all_x,
                                                        all_y,
                                                        train_size=0.8)
    #机器学习算法：KNN
    knum = k  # 设置K值
    model = KNeighborsClassifier(n_neighbors = knum)
    # 训练模型
    model.fit(train_X, train_y)
    # 预测并评估算法
    accuracy = model.score(test_X,test_y )
    print("KNN Aucc：", accuracy)

def DesTree(all_x,all_y):
    train_X, test_X, train_y, test_y = train_test_split(all_x,
                                                        all_y,
                                                        train_size=0.8)
    #机器学习算法：决策树
    model = DecisionTreeClassifier()
    # 训练模型
    model.fit(train_X, train_y)
    # 预测并评估算法
    accuracy = model.score(test_X,test_y )
    print("DeTree Aucc：", accuracy)


if __name__=="__main__":
    all_x,all_y = getData()
    LogReg(all_x,all_y,2000)
    KNN(all_x,all_y,5)
    DesTree(all_x,all_y)








