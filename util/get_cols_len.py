# -*- coding: UTF-8 -*-

def get_cols_len(in_file):
    with open(in_file, "r") as f:
        line = next(f)
        item = line.strip().split(",")
        print(len(item))


if __name__ == '__main__':
    get_cols_len("../data/train_file.txt")
