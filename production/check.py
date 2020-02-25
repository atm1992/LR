# -*- coding: UTF-8 -*-

"""
使用测试文件来检查LR模型的性能
"""
import operator
import sys

import numpy as np
from sklearn.externals import joblib
import math

sys.path.append("../")
import util.get_feature_num as GF

FEATURE_NUM = GF.get_feature_num("../data/feature_num")

def get_test_data(test_file):
    """

    :param test_file: file to check performance
    :return: two np array: test_feature, test_label
    """
    total_feature_num = FEATURE_NUM
    test_label = np.genfromtxt(test_file, dtype=np.int32, delimiter=",", usecols=-1)
    test_feature_list = list(range(total_feature_num))
    test_feature = np.genfromtxt(test_file, dtype=np.int32, delimiter=",", usecols=test_feature_list)
    return test_feature, test_label


def predict_by_lr_model(test_feature, lr_model):
    # 保存每个样本模型预测为1的概率
    result_list = []
    # prob_list 是一个N行两列的二维数组，N为测试样本的个数
    # 两列 —— 第一列的值为模型预测为0的概率；第二列的值为模型预测为1的概率
    prob_list = lr_model.predict_proba(test_feature)
    for i in range(len(prob_list)):
        result_list.append(prob_list[i][1])
    return result_list


def predict_by_lr_coef(test_feature, lr_coef):
    """
    对于参数化的model文件，其原理为 与每一个测试样本的特征相乘，然后过一遍激活函数sigmoid，便可得到输出结果
    :param test_feature:
    :param lr_coef:
    :return:
    """
    # 在NumPy中, universal functions可以对np.array中的每一个元素进行操作
    # 使用np.frompyfunc可以将任意的Python函数转换为universal functions
    # 此处是将sigmoid转化为universal functions，后面的两个1分别表示输入参数的个数 和 输出结果的个数
    sigmoid_ufunc = np.frompyfunc(sigmoid, 1, 1)
    # np.dot得到一个np.array
    return sigmoid_ufunc(np.dot(test_feature, lr_coef))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_auc(predict_list, test_label):
    """
    auc = (sum(pos_idx) - n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    sum(pos_idx) —— 先按照预测概率将所有样本(包括负样本)进行升序，将所有的正样本所处的位置index累加求和，预测概率最低的那个样本的index为1
    n_pos 为正样本的个数；n_neg 为负样本的个数
    :param predict_list:
    :param test_label:
    :return:
    """
    total_list = []
    for i in range(len(predict_list)):
        predict_score = predict_list[i]
        label = test_label[i]
        total_list.append((label, predict_score))
    n_pos, n_neg = 0, 0
    count = 1
    total_pos_idx = 0
    for label, _ in sorted(total_list, key=operator.itemgetter(1)):
        if label == 0:
            n_neg += 1
        else:
            n_pos += 1
            total_pos_idx += count
        count += 1
    auc_score = (total_pos_idx - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print("auc:{:.5f}".format(auc_score))


def get_accuracy(predict_list, test_label):
    score_thr = 0.6
    acc_num = 0
    for i in range(len(predict_list)):
        predict_score = predict_list[i]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[i]:
            acc_num += 1
    acc_score = acc_num / len(predict_list)
    print("accuracy:{:.5f}".format(acc_score))


def run_check_core(test_feature, test_label, model, score_func):
    """

    :param test_feature:
    :param test_label:
    :param model: lr_coef, lr_model
    :param score_func: use different model to predict
    """
    predict_list = score_func(test_feature, model)
    # print(predict_list[:10])
    get_auc(predict_list, test_label)
    get_accuracy(predict_list, test_label)


def run_check(test_file, lr_coef_file, lr_model_file):
    """

    :param test_file: file to check performance
    :param lr_coef_file: w1, w2, ...
    :param lr_model_file: dump file
    """
    test_feature, test_label = get_test_data(test_file)
    lr_coef = np.genfromtxt(lr_coef_file, dtype=np.float32, delimiter=",")
    lr_model = joblib.load(lr_model_file)
    # 这两种方式下计算的AUC是一致的；而得到的accuracy结果差异较大，lr_model好于lr_coef
    print("predict_by_lr_model:")
    run_check_core(test_feature, test_label, lr_model, predict_by_lr_model)
    print("predict_by_lr_coef:")
    run_check_core(test_feature, test_label, lr_coef, predict_by_lr_coef)


if __name__ == '__main__':
    # run_check("../data/test_file.txt", "../data/lr_coef", "../data/lr_model_file")
    if len(sys.argv) < 4:
        print("usage: python xx.py test_data lr_coef_file lr_model_file")
        sys.exit()

    test_file = sys.argv[1]
    lr_coef_file = sys.argv[2]
    lr_model_file = sys.argv[3]
    run_check(test_file, lr_coef_file, lr_model_file)