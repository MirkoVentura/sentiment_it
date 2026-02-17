"""Evaluation script for the SENTIPOLC 2016 shared task."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

VERBOSE = True


def main():
    argparser = ArgumentParser(description="SENTIPOLC 2016 evaluation")
    argparser.add_argument("-r", dest="result_file", action="store", help="CSV file of the run results")
    argparser.add_argument(
        "-g", dest="gold_file", action="store", default=str(DATA_DIR / "sentipolc16_gold2000.csv"),
        help="Gold standard annotation CSV file",
    )
    argparser.add_argument("-t", dest="task_name", action="store", default="res", help="Result txt file name")

    args = argparser.parse_args()
    lines = []

    # Read gold standard and populate the count matrix
    gold = {}
    gold_counts = {
        "subj": {"0": 0, "1": 0},
        "opos": {"0": 0, "1": 0},
        "oneg": {"0": 0, "1": 0},
        "iro": {"0": 0, "1": 0},
        "lpos": {"0": 0, "1": 0},
        "lneg": {"0": 0, "1": 0},
    }

    with open(args.gold_file) as f:
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) > 6:
                id_, subj, opos, oneg, iro, lpos, lneg, top = (x[1:-1] for x in parts)
            else:
                id_, subj, opos, oneg, iro, top = (x[1:-1] for x in parts)
                lpos, lneg = "0", "0"

            gold[id_] = {"subj": subj, "opos": opos, "oneg": oneg, "iro": iro, "lpos": lpos, "lneg": lneg}
            gold_counts["subj"][subj] += 1
            gold_counts["opos"][opos] += 1
            gold_counts["oneg"][oneg] += 1
            gold_counts["iro"][iro] += 1

            if len(parts) > 6:
                gold_counts["lpos"][lpos] += 1
                gold_counts["lneg"][lneg] += 1

    # Read result data
    result = {}
    with open(args.result_file) as f:
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) > 6:
                id_, subj, opos, oneg, iro, lpos, lneg, top = (x[1:-1] for x in parts)
            else:
                id_, subj, opos, oneg, iro, top = (x[1:-1] for x in parts)
            result[id_] = {"subj": subj, "opos": opos, "oneg": oneg, "iro": iro}

    # Evaluation: single classes
    for task in ["subj", "opos", "oneg", "iro"]:
        if VERBOSE:
            print(f"\ntask: {task}")
        lines.append(f"\ntask: {task}")

        header = "prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc."
        if VERBOSE:
            print(header)
        lines.append(header)

        correct = {"0": 0, "1": 0}
        assigned = {"0": 0, "1": 0}
        precision = {"0": 0.0, "1": 0.0}
        recall = {"0": 0.0, "1": 0.0}
        fscore = {"0": 0.0, "1": 0.0}

        for id_, gold_labels in gold.items():
            if id_ not in result or result[id_][task] == "":
                pass
            else:
                assigned[result[id_][task]] += 1
                if gold_labels[task] == result[id_][task]:
                    correct[result[id_][task]] += 1

        for label in ["0", "1"]:
            try:
                precision[label] = float(correct[label]) / float(assigned[label])
                recall[label] = float(correct[label]) / float(gold_counts[task][label])
                fscore[label] = (2.0 * precision[label] * recall[label]) / (precision[label] + recall[label])
            except (ZeroDivisionError, KeyError):
                fscore[label] = 0.0

        row = (
            f"{precision['0']:.4f}\t{recall['0']:.4f}\t{fscore['0']:.4f}\t"
            f"{precision['1']:.4f}\t{recall['1']:.4f}\t{fscore['1']:.4f}\t"
            f"{(fscore['0'] + fscore['1']) / 2.0:.4f}"
        )
        print(row)
        lines.append(row)

    # Polarity evaluation
    if VERBOSE:
        print("\ntask: polarity")
    lines.append("\ntask: polarity")

    if VERBOSE:
        print("Combined F-score")
    lines.append("Combined F-score")

    correct = {"opos": {"0": 0, "1": 0}, "oneg": {"0": 0, "1": 0}}
    assigned = {"opos": {"0": 0, "1": 0}, "oneg": {"0": 0, "1": 0}}
    precision = {"opos": {"0": 0.0, "1": 0.0}, "oneg": {"0": 0.0, "1": 0.0}}
    recall = {"opos": {"0": 0.0, "1": 0.0}, "oneg": {"0": 0.0, "1": 0.0}}
    fscore = {"opos": {"0": 0.0, "1": 0.0}, "oneg": {"0": 0.0, "1": 0.0}}

    for id_, gold_labels in gold.items():
        for cl in ["opos", "oneg"]:
            if id_ not in result or result[id_][cl] == "":
                pass
            else:
                assigned[cl][result[id_][cl]] += 1
                if gold_labels[cl] == result[id_][cl]:
                    correct[cl][result[id_][cl]] += 1

    for cl in ["opos", "oneg"]:
        for label in ["0", "1"]:
            try:
                precision[cl][label] = float(correct[cl][label]) / float(assigned[cl][label])
                recall[cl][label] = float(correct[cl][label]) / float(gold_counts[cl][label])
                fscore[cl][label] = (2.0 * precision[cl][label] * recall[cl][label]) / (
                    precision[cl][label] + recall[cl][label]
                )
            except (ZeroDivisionError, KeyError):
                fscore[cl][label] = 0.0

    fscore_pos = (fscore["opos"]["0"] + fscore["opos"]["1"]) / 2.0
    fscore_neg = (fscore["oneg"]["0"] + fscore["oneg"]["1"]) / 2.0

    lines.append(f"{(fscore_pos + fscore_neg) / 2.0:.4f}")

    with open(f"./{args.task_name}.txt", "w") as txt_file:
        for line in lines:
            txt_file.write(line + " \n")


if __name__ == "__main__":
    main()
