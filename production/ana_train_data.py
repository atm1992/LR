# -*- coding: UTF-8 -*-

"""
feature selection and data selection
"""
import operator
import sys

import pandas as pd
import numpy as np

pd.options.display.max_columns = 30
pd.options.display.max_colwidth = 150


def get_input(in_train_file, in_test_file):
    """
    返回训练集和测试集
    :param in_train_file:
    :param in_test_file:
    :return:
        pd.DataFrame train_data
        pd.DataFrame test_data
    """
    dtype_dict = {
        "age": np.int32,
        "education-num": np.int32,
        "capital-gain": np.int32,
        "capital-loss": np.int32,
        "hours-per-week": np.int32
    }
    # 总共有15个特征，其中第3个特征(fnlwgt)是不需要的
    use_list = list(range(15))
    use_list.remove(2)
    # 第一行为列名（header=0）；na_values缺省值
    train_data_df = pd.read_csv(in_train_file, sep=',', header=0, dtype=dtype_dict, na_values="?",
                                usecols=use_list)
    # 若某行中含有空值，则直接删除该行
    train_data_df = train_data_df.dropna(axis=0, how="any")

    # 第一行为列名（header=0）；na_values缺省值
    test_data_df = pd.read_csv(in_test_file, sep=',', header=0, dtype=dtype_dict, na_values="?",
                               usecols=use_list)
    # 若某行中含有空值，则直接删除该行
    test_data_df = test_data_df.dropna(axis=0, how="any")
    return train_data_df, test_data_df


def label_trans(x):
    if x == ">50K":
        return "1"
    else:
        return "0"


def process_label_feature(label_feature_str, df_in):
    """

    :param label_feature_str: "label"
    :param df_in: input DataFrame
    """
    df_in.loc[:, label_feature_str] = df_in.loc[:, label_feature_str].apply(label_trans)


def dict_trans(dict_in):
    """

    :param dict_in: key str, value int
    :return: a dict, key str, value index for example 0,1,2
    """
    out_dict = {}
    index = 0
    # 该离散值特征列中出现次数count最高的item所对应的离散化特征的第一位为1，其余位为0。哈夫曼树
    for item, count in sorted(dict_in.items(), key=operator.itemgetter(1), reverse=True):
        out_dict[item] = index
        index += 1
    return out_dict


def discrete_to_feature(x, feature_dict):
    """

    :param x: element
    :param feature_dict: pos dict
    :return: a str, as "0,1,0"
    """
    # len(feature_dict)表示特征维度
    out_list = [0] * len(feature_dict)
    if x in feature_dict:
        out_list[feature_dict[x]] = 1
    return ",".join([str(ele) for ele in out_list])


def process_discrete_feature(feature_str, df_train, df_test):
    """
    返回离散型特征的维度，即 该特征的每一个值用多少位来表示，只有一位为1，其余位为0
    :param feature_str: label_in, 离散特征
    :param df_train: train_data_df
    :param df_test: test_data_df
    :return: the dim of the output feature
    """
    origin_dict = df_train.loc[:, feature_str].value_counts().to_dict()
    feature_dict = dict_trans(origin_dict)
    # 该特征的每一个值都转换为只有一位为1，其余位为0的离散化特征(字符串形式)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(discrete_to_feature, args=(feature_dict,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(discrete_to_feature, args=(feature_dict,))
    # print(df_train.loc[:3, feature_str])
    # print(feature_dict)
    return len(feature_dict)


def list_trans(in_dict):
    """
    使用一个数组返回输入dict中'min','25%','50%','75%','max'这些字段的值
    :param in_dict: {'count': 30162.0, 'mean': 38.437901995888865, 'std': 13.134664776856338, 'min': 17.0, '25%': 28.0, '50%': 37.0, '75%': 47.0, 'max': 90.0}
    :return: a list, [0.1,0.2,0.3,0.4,0.5]
    """
    out_list = [0] * 5
    key_list = ['min', '25%', '50%', '75%', 'max']
    for i in range(len(key_list)):
        fix_key = key_list[i]
        if fix_key not in in_dict:
            print("error")
            sys.exit()
        else:
            out_list[i] = in_dict[fix_key]
    return out_list


def continuous_to_feature(x, feature_list):
    """

    :param x: element
    :param feature_list: list for feature trans
    :return: a str, as "0,1,0"
    """
    # 根据连续特征的'min','25%','50%','75%','max'这5个值，将该连续特征离散化成4段
    feature_len = len(feature_list) - 1
    result = [0] * feature_len
    for i in range(feature_len):
        if feature_list[i] <= x < feature_list[i + 1]:
            result[i] = 1
            break
    return ",".join([str(ele) for ele in result])


def process_continuous_feature(feature_str, df_train, df_test):
    """
    返回连续型特征的维度，即 该特征的每一个值用多少位来表示，只有一位为1，其余位为0
    :param feature_str: label_in, 连续特征
    :param df_train: train_data_df
    :param df_test: test_data_df
    :return: the dim of the output feature
    """
    origin_dict = df_train.loc[:, feature_str].describe().to_dict()
    # 得到指定连续特征feature_str的统计值'min','25%','50%','75%','max'
    feature_list = list_trans(origin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(continuous_to_feature, args=(feature_list,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(continuous_to_feature, args=(feature_list,))
    return len(feature_list) - 1


def out_file(df_in, out_file):
    with open(out_file, "w+") as f:
        for row_idx in df_in.index:
            out_line = ",".join([str(ele) for ele in df_in.loc[row_idx].values])
            f.write(out_line + "\n")


def add(str_1, str_2):
    """

    :param str_1: as "0,0,1,0"
    :param str_2: as "1,0,0,0"
    :return: as "0,0,1,0,0,0,0"
    """
    list_1 = str_1.split(",")
    list_2 = str_2.split(",")
    list_1_len = len(list_1)
    list_2_len = len(list_2)
    return_list = [0] * (list_1_len * list_2_len)
    try:
        idx_1 = list_1.index("1")
    except:
        idx_1 = 0
    try:
        idx_2 = list_2.index("1")
    except:
        idx_2 = 0
    # idx_1 * list_2_len + idx_2 这个计算公式是根据需求自定义的
    return_list[idx_1 * list_2_len + idx_2] = 1
    return ",".join([str(ele) for ele in return_list])


def combine_feature(feature_1, feature_2, new_feature, train_data_df, test_data_df, feature_num_dict):
    """
    将两个特征进行组合
    :param feature_1:
    :param feature_2:
    :param new_feature: combine feature name
    :param train_data_df:
    :param test_data_df:
    :param feature_num_dict: dim of every feature. key feature name, value len of the dim
    :return: the dim of the new feature
    """
    train_data_df[new_feature] = train_data_df.apply(lambda row: add(row[feature_1], row[feature_2]), axis=1)
    test_data_df[new_feature] = test_data_df.apply(lambda row: add(row[feature_1], row[feature_2]), axis=1)
    if feature_1 not in feature_num_dict:
        print("error")
        sys.exit()
    if feature_2 not in feature_num_dict:
        print("error")
        sys.exit()
    return feature_num_dict[feature_1] * feature_num_dict[feature_2]


def ana_train_data(in_train_file, in_test_file, out_train_file, out_test_file, feature_num_file):
    """
    
    :param in_train_file: 
    :param in_test_file: 
    :param out_train_file: 
    :param out_test_file:
    :param feature_num_file:
    :return: 
    """
    train_data_df, test_data_df = get_input(in_train_file, in_test_file)
    discrete_feature_list = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                             "native-country"]
    continuous_feature_list = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    index_list = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "relationship",
                  "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    process_label_feature("label", train_data_df)
    process_label_feature("label", test_data_df)
    # 所有离散特征总的维度
    discrete_feature_num = 0
    # 所有连续特征总的维度
    continuous_feature_num = 0
    # 存储每一个特征及其维度。组合特征的维度为子特征维度的乘积
    feature_num_dict = {}
    for discrete_feature in discrete_feature_list:
        tmp_feature_num = process_discrete_feature(discrete_feature, train_data_df, test_data_df)
        discrete_feature_num += tmp_feature_num
        feature_num_dict[discrete_feature] = tmp_feature_num
    for continuous_feature in continuous_feature_list:
        tmp_feature_num = process_continuous_feature(continuous_feature, train_data_df, test_data_df)
        continuous_feature_num += tmp_feature_num
        feature_num_dict[continuous_feature] = tmp_feature_num

    new_feature_len_1 = combine_feature("age", "capital-gain", "age_gain", train_data_df, test_data_df,
                                        feature_num_dict)
    new_feature_len_2 = combine_feature("capital-gain", "capital-loss", "gain_loss", train_data_df, test_data_df,
                                        feature_num_dict)
    train_data_df = train_data_df.reindex(columns=index_list + ["age_gain", "gain_loss", "label"])
    test_data_df = test_data_df.reindex(columns=index_list + ["age_gain", "gain_loss", "label"])
    # print(train_data_df["age"][:2])
    # print(train_data_df["occupation"][:2])
    # print(train_data_df["age_occupation"][:2])
    out_file(train_data_df, out_train_file)
    out_file(test_data_df, out_test_file)

    with open(feature_num_file, "w+") as f:
        f.write("feature_num:{}".format(
            str(discrete_feature_num + continuous_feature_num + new_feature_len_1 + new_feature_len_2)))


if __name__ == '__main__':
    # ana_train_data("../data/adult_train.txt", "../data/adult_test_2.txt", "../data/train_file.txt", "../data/test_file.txt", "../data/feature_num")
    if len(sys.argv) < 6:
        print("usage: python xx.py origin_train_data origin_test_data train_data test_data feature_num_file")
        sys.exit()

    in_train_file = sys.argv[1]
    in_test_file = sys.argv[2]
    out_train_file = sys.argv[3]
    out_test_file = sys.argv[4]
    feature_num_file = sys.argv[5]
    ana_train_data(in_train_file, in_test_file, out_train_file, out_test_file, feature_num_file)
