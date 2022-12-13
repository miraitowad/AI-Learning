from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
from functools import reduce
 
 
# 导入数据集切割训练与测试数据
 
data = load_digits()
data_D = preprocessing.StandardScaler().fit_transform(data.data)
data_L = data.target
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,random_state=1,test_size=0.7)
 
def SelectModel(modelname):
 
    if modelname == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
 
    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
 
    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
 
    elif modelname == "XGBOOST":
        import xgboost as xgb
        model = xgb()
 
    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
    else:
        pass
    return model
 
def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds,random_state=1)
    oof_train = np.zeros((ntrain,classnum))
    oof_test = np.zeros((ntest,classnum))
 
 
    for i,(train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index] # 数据
        kf_y_train = y_train[train_index] # 标签
 
        kf_X_test = X_train[test_index]  # k-fold的验证集
 
        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)
 
        oof_test += clf.predict_proba(X_test)
    oof_test = oof_test/float(n_folds)
    return oof_train, oof_test
 
# 单纯使用一个分类器的时候
clf_second = RandomForestClassifier()
clf_second.fit(data_train, label_train)
pred = clf_second.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100

# 91.0969793323
 
# 使用stacking方法的时候
# 第一级，重构特征当做第二级的训练集
modelist = ['SVM','GBDT','RF','KNN']
newfeature_list = []
newtestdata_list = []
for modelname in modelist:
    clf_first = SelectModel(modelname)
    oof_train_ ,oof_test_= get_oof(clf=clf_first,n_folds=10,X_train=data_train,y_train=label_train,X_test=data_test)
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)
 
# 特征组合
newfeature = reduce(lambda x,y:np.concatenate((x,y),axis=1),newfeature_list)    
newtestdata = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list)
 
 
# 第二级，使用上一级输出的当做训练集
clf_second1 = RandomForestClassifier()
clf_second1.fit(newfeature, label_train)
pred = clf_second1.predict(newtestdata)
accuracy = metrics.accuracy_score(label_test, pred)*100
