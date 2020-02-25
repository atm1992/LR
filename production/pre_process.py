# -*- coding: UTF-8 -*-

import pandas as pd

pd.options.display.max_columns = 20


def convert(x, col):
    x = str(x).strip()
    if col == "label":
        x = x.rstrip('.')
    return x


def pre_process_test(in_file, out_file):
    df = pd.read_csv(in_file, header=0)
    for col in df.columns:
        df[col] = df[col].apply(convert, args=(col,))
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    pre_process_test("../data/adult_test.txt", "../data/adult_test_2.txt")
