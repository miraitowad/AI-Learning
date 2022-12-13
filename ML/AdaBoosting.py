from numpy import *

def loadSimpData():
    """
    构造一个简单数据集
    :return: 数据集，标签
    """
    dataMat = matrix([[1.0,2.1],
                      [2.,1.1],
                      [1.3,1.],
                      [1.,1.],
                      [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    通过阈值比较对数据进行分类
    :param dataMatrix: 数据集
    :param dimen: 维度
    :param threshVal: 阈值
    :param threshIneq: 不等号
    :return:类别估计数组
    """
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':  # lt:表示小于等于号
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    :param dataArr:数据集
    :param classLabels:标签
    :param D:权重向量
    :return:最佳单层决策树相关信息字典、错误率、类别估计值
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}  # 存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestClasEst = mat(zeros((m,1)))  # 最佳类别估计值
    minError = inf  #初始化最小错误率为正无穷大
    for i in range(n):  # 循环所有特征
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):  # 循环每个步长
            for inequal in ['lt','gt']:  # 循环每个不等号，即大于和小于之间切换
                # 阈值可以设置为整个取值范围之外
                threshVal = (rangeMin+float(j)*stepSize)  # 阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) # 预测的分类结果
                errArr = mat(ones((m,1)))  # 错误向量
                errArr[predictedVals==labelMat] = 0  # 预测值和真实值相等时，设置为0
                weightedError = D.T*errArr  # 权重错误率
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %(i,threshVal,inequal,weightedError))
                # 当权重错误率小于最小错误率时，更新最小错误率、最佳决策树相关信息、类别估计值
                if weightedError<minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i  # 维度
                    bestStump['thresh'] = threshVal # 阈值
                    bestStump['ineq'] = inequal  # 不等号
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    基于单层决策树的AdaBoost训练过程
    :param dataArr: 数据集
    :param classLabels: 标签
    :param numIt: 迭代次数
    :return:多个弱分类器，包含其对应的alpha值
    """
    weakClassArr = [] # 单层决策树数组
    m = shape(dataArr)[0]
    # D为每个数据点的权重，每个数据点的权重都会被初始化为1/m
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))  # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        # 构建一个最佳单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D: ",D.T)
        # max(error,1e-16)))用于确保在没有错误时不会发生除零溢出
        # alpha:本次单层决策树输出结果的权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  # 计算alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump) # 将最佳单层决策树存储到单层决策树数组中
        print("classEst: ",classEst.T)
        # 更新数据样本权值D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        # 更新累计类别估计值
        aggClassEst +=alpha*classEst
        print ("aggClassEst: ",aggClassEst.T)
        # 计算错误率
        aggErrors = multiply(sign(aggClassEst)!= mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        # 如果错误率为0.0，则退出循环
        if errorRate==0.0:
            break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    """
    利用训练出的多个弱分类器，在测试集上进行分类预测
    :param datToClass:一个或多个待分类样例
    :param classifierArr:多个弱分类器组成的数组
    :return:预测的类别值
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))  # 类别估计值
    for i in range(len(classifierArr)):  # 遍历每个分类器
        # 对每个分类器得到一个类别估计值
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst) # 如果aggClassEst>0,则返回1，如果aggClassEst<=0,则返回-1

if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    D = mat(ones((5,1))/5)
    classifierArr = adaBoostTrainDS(dataMat,classLabels,9)  # 得到弱分类器数组
    print("预测[0,0]的类别结果:\n",adaClassify([0,0],classifierArr))
    print("预测[[5,5],[0, 0]]的类别结果:\n",adaClassify([[5,5],[0, 0]], classifierArr))