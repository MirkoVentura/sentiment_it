"""Combine separate positive/negative prediction files into a single evaluation file."""

from __future__ import annotations

import argparse
import csv

import pandas as pd


def find_predict_by_id(row, data_pred, task):
    val = data_pred[data_pred["idtwitter"] == row.idtwitter]
    print(val)
    if task == "pos":
        return val["pred_pos"].values[0]
    if task == "neg":
        return val["pred_neg"].values[0]
    return -1


def main():
    parser = argparse.ArgumentParser(description="Combine opos/oneg predictions")
    parser.add_argument("--gold", type=str, help="Gold standard file", default="./data/test_gold.csv")
    parser.add_argument("--predictPos", type=str, help="Positive predictions file", default="./res_bert_try_opos.csv")
    parser.add_argument("--predictNeg", type=str, help="Negative predictions file", default="./res_bert_try_oneg.csv")
    parser.add_argument("--runName", type=str, help="Run name identifier", default="001")

    args = parser.parse_args()

    data_gold = pd.read_csv(args.gold, sep="\t")
    data_pred = pd.read_csv(args.predictPos, sep="|")
    data_pred.pred_pos.astype(str)

    print(data_pred.head())

    data_pred_neg = pd.read_csv(args.predictNeg, sep="|")
    data_pred_neg.pred_neg.astype(str)

    data_gold["pred_pos"] = data_gold.apply(lambda row: find_predict_by_id(row, data_pred, "pos"), axis=1)
    data_gold["pred_neg"] = data_gold.apply(lambda row: find_predict_by_id(row, data_pred_neg, "neg"), axis=1)

    print(data_gold.head())

    data_to_save = data_gold[["idtwitter", "subj", "pred_pos", "pred_neg", "iro", "lpos", "lneg", "top"]]
    data_to_save = data_to_save.astype(str)
    data_to_save.to_csv(
        f"./res_bert_to_evaluete_{args.runName}.csv",
        header=None,
        index=None,
        sep=",",
        mode="a",
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )


if __name__ == "__main__":
    main()
