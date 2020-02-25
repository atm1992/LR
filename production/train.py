# -*- coding: UTF-8 -*-
import sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.externals import joblib

sys.path.append("../")
import util.get_feature_num as GF

FEATURE_NUM = GF.get_feature_num("../data/feature_num")


def train_lr_model(train_file, model_coef, model_file):
    """

    :param train_file: process file for lr training
    :param model_coef: w1, w2, ...
    :param model_file: model pkl
    """
    # 98+20=118. 所有离散特征的总维度为98，所有连续特征的总维度为20
    # 118 表示所有特征的总维度。label的维度为1，因此train_file.txt、test_file.txt的列数为119
    total_feature_num = FEATURE_NUM
    # usecols=-1 表示使用最后一列， 也就是label
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = list(range(total_feature_num))
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=feature_list)

    # Cs - 正则化参数，分别为 1、0.1、0.01；penalty="l2" L2正则；tol 参数迭代停止的条件；max_iter 最大迭代次数；
    # cv=5 5折交叉验证，将训练数据分为5份，每次拿80%作为训练集，20%作为测试集，总共进行5次训练
    # 最优化方法solver选择拟牛顿法
    # 由结果可知，正则化参数为1时，模型的AUC最高。这里将[1, 10, 100]改为[1]
    lr_cf = LRCV(Cs=[1], penalty="l2", tol=1e-4, max_iter=500, cv=5).fit(train_feature, train_label)
    # 得到一个5行3列的二维数组
    scores = list(lr_cf.scores_.values())[0]
    # 查看每一个正则化参数(1、0.1、0.01)所对应的5折交叉验证的平均准确率
    # 由结果可知，正则化参数为0.01时，模型的准确率最高
    print("diff:", ",".join([str(ele) for ele in scores.mean(axis=0)]))
    # 各个正则化参数的总平均准确率
    print("accuracy:{0} (+-{1:.3f})".format(scores.mean(), scores.std() * 2))

    # 除了上述查看模型的准确率，还需要关心模型的AUC，scoring="roc_auc"
    lr_cf = LRCV(Cs=[1], penalty="l2", tol=1e-4, max_iter=500, cv=5, scoring="roc_auc").fit(train_feature, train_label)
    scores = list(lr_cf.scores_.values())[0]
    # 由结果可知，正则化参数为1时，模型的AUC最高。与上述的准确率最优参数不一致，这里倾向于选择AUC最高的参数1
    print("diff:", ",".join([str(ele) for ele in scores.mean(axis=0)]))
    # 各个正则化参数的总平均AUC
    print("AUC:{0} (+-{1:.3f})".format(scores.mean(), scores.std() * 2))

    coef = lr_cf.coef_[0]
    # 保存模型的训练参数
    with open(model_coef, "w+") as f:
        f.write(",".join([str(ele) for ele in coef]))

    # 将整个模型实例化到文件
    joblib.dump(lr_cf, model_file)


if __name__ == '__main__':
    # train_lr_model("../data/train_file.txt", "../data/lr_coef", "../data/lr_model_file")
    if len(sys.argv) < 4:
        print("usage: python xx.py train_data lr_coef_file lr_model_file")
        sys.exit()

    train_file = sys.argv[1]
    model_coef = sys.argv[2]
    model_file = sys.argv[3]
    train_lr_model(train_file, model_coef, model_file)
