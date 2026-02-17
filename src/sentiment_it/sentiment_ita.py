"""Italian sentiment analysis using BERT models (SENTIPOLC 2016 tasks)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import wordninja
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from simpletransformers.classification import ClassificationModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

ita_moji = pd.read_csv(DATA_DIR / "italianMoji.csv", sep=";")
lm = wordninja.LanguageModel(str(DATA_DIR / "words.last_all.txt.gz"))

EMOTICON_IT = {
    ":‑)": "felice", ": ‑)": "felice", ": ‑ )": "felice",
    ":-]": "felice", ": - ]": "felice", ": -]": "felice",
    ":-3": "felice", ": - 3": "felice", ": -3": "felice",
    ":->": "felice", ": - >": "felice", ": ->": "felice",
    "8-)": "felice", "8 -)": "felice", "8 - )": "felice",
    ":-}": "felice", ": - }": "felice", ": -}": "felice",
    ":)": "felice", ": )": "felice",
    ":]": "felice", ": ]": "felice",
    ":3": "felice", ": 3": "felice",
    ":>": "felice", ": >": "felice",
    "8)": "felice", "8 )": "felice",
    ":}": "felice", ": }": "felice",
    ":o)": "felice", ":o )": "felice", ": o )": "felice",
    ":c)": "felice", ": c )": "felice", ":c )": "felice",
    ":^)": "felice", ": ^ )": "felice", ": ^)": "felice",
    "=]": "felice", "= ]": "felice",
    "=)": "felice", "= )": "felice",
    ":-))": "felice", ": - ) )": "felice",
    ":- ) )": "felice", ":- ))": "felice", ": -))": "felice",
    ":‑D": "felice", ": ‑ D": "felice", ": ‑D": "felice",
    "8‑D": "felice", "8 ‑D": "felice", "8 ‑ D": "felice",
    "x‑D": "felice", "x ‑ D": "felice", "x ‑D": "felice",
    "X‑D": "felice", "X ‑ D": "felice", "X ‑D": "felice",
    ":D": "felice", ": D": "felice",
    "8D": "felice", "8 D": "felice",
    "xD": "felice", "x D": "felice",
    "XD": "felice", "X D": "felice",
    ":‑(": "triste", ": ‑(": "triste", ": ‑ (": "triste",
    ":‑c": "triste", ": ‑c": "triste",
    ":‑<": "triste", ": ‑ <": "triste",
    ":‑[": "triste", ": ‑ [": "triste",
    ":(": "triste", ": (": "triste",
    ":c": "triste", ": c": "triste",
    ":<": "triste", ": <": "triste",
    ":[": "triste", ": [": "triste",
    ":-||": "triste", ": - | |": "triste",
    ": - ||": "triste", ": -||": "triste", ": -| |": "triste",
    ">:[": "triste", ">: [": "triste", "> : [": "triste",
    ":{": "triste", ": {": "triste",
    ":@": "triste", ": @": "triste",
    ">:(": "triste", "> : (": "triste",
    ":'‑(": "triste", ": '‑(": "triste",
    ": ' ‑(": "triste", ": ' ‑ (": "triste",
    ":'(": "triste", ": ' (": "triste", ": '(": "triste",
    ":‑P": "scherzoso", ": ‑P": "scherzoso", ": ‑ P": "scherzoso",
    "X‑P": "scherzoso", "X ‑ P": "scherzoso", "X ‑P": "scherzoso",
    "x‑p": "scherzoso", "x ‑p": "scherzoso",
    ":‑p": "scherzoso", ": ‑p": "scherzoso", ": ‑ p": "scherzoso",
    ":‑Þ": "scherzoso", ": ‑ Þ": "scherzoso",
    ":‑þ": "scherzoso", ": ‑þ": "scherzoso",
    ":‑b": "scherzoso", ": ‑ b": "scherzoso", ": ‑b": "scherzoso",
    ":P": "scherzoso", ": P": "scherzoso",
    "XP": "scherzoso", "X P": "scherzoso",
    "xp": "scherzoso", "x p": "scherzoso",
    ":p": "scherzoso", ": p": "scherzoso",
    ":Þ": "scherzoso", ": Þ": "scherzoso",
    ":þ": "scherzoso", ": þ": "scherzoso",
    ":b": "scherzoso", ": b": "scherzoso",
    "<3": "amore", "< 3": "amore",
    ":*": "amore", ": *": "amore",
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _str_to_bool(value: str) -> bool:
    """Parse boolean CLI arguments correctly."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def add_pred_pos(row, model, task):
    predictions, _raw_outputs = model.predict([row.text_preprocessed])
    return predictions[0]


def trainer(train_df, output_dir, preproc, args):
    conf_path = Path(args.modelConf)
    if not conf_path.is_absolute():
        conf_path = DATA_DIR / conf_path.name

    with open(conf_path) as f:
        model_param = json.loads(f.read())
    model_param["output_dir"] = output_dir

    print(model_param)

    model_name = "dbmdz/bert-base-italian-xxl-uncased"
    if preproc == "alberto":
        model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"

    model = ClassificationModel("bert", model_name, args=model_param)
    model.train_model(train_df)
    return model


def tester(model, task, test_df, preproc, run_name):
    filename = f"./res_bert_try_{preproc}_{run_name}"
    if task == "opos":
        filename += "_opos.csv"
        test_df["pred_pos"] = test_df.apply(lambda row: add_pred_pos(row, model, task), axis=1)
    else:
        filename += "_oneg.csv"
        test_df["pred_neg"] = test_df.apply(lambda row: add_pred_pos(row, model, task), axis=1)

    test_df.to_csv(filename, index=None, sep="|", mode="a", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


def alberto_preprocessing(row, text_processor):
    s = row.text.lower()
    s = " ".join(text_processor.pre_process_doc(s))
    s = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00011000-\U0011ffff]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(\w)\1{2,}", r"\1\1", s)
    s = s.strip()
    return s


def mirko_preprocessing(row, args, text_processor):
    tweet = row.text

    if args.normalizeNoise:
        tweet = " ".join(text_processor.pre_process_doc(tweet))

    if args.doEmoticon:
        words = tweet.split()
        tweet = " ".join(EMOTICON_IT.get(w, w) for w in words)

    if args.removeEmoticon:
        words = tweet.split()
        tweet = " ".join(" " if w in EMOTICON_IT else w for w in words)

    if args.normalizeEmoticon:
        words = tweet.split()
        tweet = " ".join("<emoticon>" if w in EMOTICON_IT else w for w in words)

    if args.doEmoji:
        if emoji.emoji_count(tweet) != 0:
            emos = emoji.emoji_list(tweet)
            for emo in emos:
                single_moji = str(emo["emoji"])
                ita_m = ita_moji[ita_moji["emoji"] == single_moji]
                if len(ita_m["text_ita"].values) != 0:
                    significato = ita_m["text_ita"].values[0]
                    tweet = tweet.replace(single_moji, significato)
                else:
                    tweet = tweet.replace(single_moji, "")

    if args.removeEmoji:
        tweet = EMOJI_PATTERN.sub(r"", tweet)

    if args.normalizeEmoji:
        tweet = EMOJI_PATTERN.sub(r" <emoji> ", tweet)

    if args.removeMention:
        tweet = " ".join(re.sub(r"(@[A-Za-z0-9]+)", " ", tweet).split())

    if args.normalizeMention:
        tweet = " ".join(re.sub(r"(@[A-Za-z0-9]+)", " @user ", tweet).split())

    if args.removeUrl:
        tweet = " ".join(re.sub(r"(\w+://\S+)", " ", tweet).split())

    if args.normalizeUrl:
        tweet = " ".join(re.sub(r"(\w+://\S+)", " url ", tweet).split())

    if not args.rawHashtag:
        elems = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]
        for elem in elems:
            translated = " ".join(lm.split(elem))
            if args.unpackHastags:
                tweet = tweet.replace(f"#{elem}", translated)
            if args.tagAndUnpackHastags:
                tweet = tweet.replace(f"#{elem}", f"<{translated}>")
            if args.removeHastags:
                tweet = tweet.replace(f"#{elem}", " ")
            if args.normalizeHastags:
                tweet = tweet.replace(f"#{elem}", "#hashtag")

    if args.removePunctuation:
        tweet = " ".join(re.sub(r"[.,!?:;\-=]", " ", tweet).split())

    tweet = re.sub(r"\s+", " ", tweet)
    return tweet.lower() if args.doLower else tweet


def build_parser():
    parser = argparse.ArgumentParser(description="Italian sentiment analysis with BERT")
    parser.add_argument("--task", type=str, help="Task: opos or oneg", default="opos")
    parser.add_argument("--odir", type=str, help="Output directory", default="./")
    parser.add_argument("--trainSet", type=str, help="Training CSV path", default=str(DATA_DIR / "training_set_sentipolc16.csv"))
    parser.add_argument("--testSet", type=str, help="Test CSV path", default=str(DATA_DIR / "test_set_sentipolc16_gold2000.csv"))
    parser.add_argument("--preproc", type=str, help="Preprocessor: alberto, mirko, or raw", default="raw")
    parser.add_argument("--doShuffle", type=_str_to_bool, help="Shuffle training data", default=True)

    parser.add_argument("--doEmoji", type=_str_to_bool, help="Translate emoji", default=True)
    parser.add_argument("--removeEmoji", type=_str_to_bool, help="Remove emoji", default=False)
    parser.add_argument("--normalizeEmoji", type=_str_to_bool, help="Normalize emoji", default=False)
    parser.add_argument("--doEmoticon", type=_str_to_bool, help="Translate emoticons", default=True)
    parser.add_argument("--removeEmoticon", type=_str_to_bool, help="Remove emoticons", default=False)
    parser.add_argument("--normalizeEmoticon", type=_str_to_bool, help="Normalize emoticons", default=False)
    parser.add_argument("--normalizeMention", type=_str_to_bool, help="Replace mentions with @user", default=False)
    parser.add_argument("--removeMention", type=_str_to_bool, help="Remove mentions", default=False)
    parser.add_argument("--normalizeUrl", type=_str_to_bool, help="Replace URLs with 'url'", default=False)
    parser.add_argument("--removeUrl", type=_str_to_bool, help="Remove URLs", default=False)
    parser.add_argument("--unpackHastags", type=_str_to_bool, help="Unpack hashtags", default=False)
    parser.add_argument("--tagAndUnpackHastags", type=_str_to_bool, help="Tag and unpack hashtags", default=False)
    parser.add_argument("--normalizeHastags", type=_str_to_bool, help="Normalize hashtags", default=False)
    parser.add_argument("--removeHastags", type=_str_to_bool, help="Remove hashtags", default=False)
    parser.add_argument("--removePunctuation", type=_str_to_bool, help="Remove punctuation", default=False)
    parser.add_argument("--normalizeNoise", type=_str_to_bool, help="Normalize noise with ekphrasis", default=False)
    parser.add_argument("--rawHashtag", type=_str_to_bool, help="Keep hashtags raw", default=False)
    parser.add_argument("--doLower", type=_str_to_bool, help="Lowercase text", default=True)
    parser.add_argument("--pureModel", type=_str_to_bool, help="Use pure AlBERTo preprocessing", default=False)

    parser.add_argument("--runName", type=str, help="Run name identifier", default="000")
    parser.add_argument("--modelConf", type=str, help="Model config path", default="conf.json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    data_train = pd.read_csv(args.trainSet, sep=",")
    data_test = pd.read_csv(args.testSet, sep=",")

    if args.doShuffle:
        data_train = data_train.reindex(np.random.permutation(data_train.index))

    preproc = args.preproc

    if preproc == "mirko":
        text_processor = TextPreProcessor(
            remove=["email", "percent", "money", "phone", "time", "date", "number"],
            annotate={},
            fix_html=True,
            unpack_hashtags=False,
            tokenizer=SocialTokenizer(lowercase=args.doLower).tokenize,
            dicts=[emoticons],
        )
        data_train["text_preprocessed"] = data_train.apply(lambda row: mirko_preprocessing(row, args, text_processor), axis=1)
        data_test["text_preprocessed"] = data_test.apply(lambda row: mirko_preprocessing(row, args, text_processor), axis=1)

    elif preproc == "alberto":
        text_processor_mirko = TextPreProcessor(
            remove=["email", "percent", "money", "phone", "time", "date", "number"],
            annotate={},
            fix_html=True,
            unpack_hashtags=False,
            tokenizer=SocialTokenizer(lowercase=args.doLower).tokenize,
            dicts=[emoticons],
        )
        text_processor = TextPreProcessor(
            remove=["url", "email", "user", "percent", "money", "phone", "time", "date", "number"],
            annotate={"hashtag"},
            fix_html=True,
            unpack_hashtags=True,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons],
        )
        if args.pureModel:
            data_train["text_preprocessed"] = data_train.apply(
                lambda row: alberto_preprocessing(row, text_processor), axis=1
            )
            data_test["text_preprocessed"] = data_test.apply(
                lambda row: alberto_preprocessing(row, text_processor), axis=1
            )
        else:
            data_train["text_preprocessed"] = data_train.apply(
                lambda row: mirko_preprocessing(row, args, text_processor_mirko), axis=1
            )
            data_test["text_preprocessed"] = data_test.apply(
                lambda row: mirko_preprocessing(row, args, text_processor_mirko), axis=1
            )

    elif preproc == "raw":
        data_train["text_preprocessed"] = data_train["text"]
        data_test["text_preprocessed"] = data_test["text"]

    pd.set_option("display.max_colwidth", 800)
    print("***** TRAIN HEAD ***")
    print(data_train.head())
    print("***** TEST HEAD ***")
    print(data_test.head())

    task = args.task
    output_dir = args.odir

    if task == "opos":
        output_dir += f"predictor_all_pos_{preproc}_{args.runName}"
        train_df = data_train[["text_preprocessed", "opos"]]
        train_df.text_preprocessed.astype(str)
        train_df.opos.astype(float)
        test_df = data_test[["idtwitter", "text_preprocessed", "opos"]]
        test_df.text_preprocessed.astype(str)
        test_df.opos.astype(float)
        model = trainer(train_df, output_dir, preproc, args)
    elif task == "oneg":
        output_dir += f"predictor_all_neg_{preproc}_{args.runName}"
        train_df = data_train[["text_preprocessed", "oneg"]]
        train_df.text_preprocessed.astype(str)
        train_df.oneg.astype(float)
        test_df = data_test[["idtwitter", "text_preprocessed", "oneg"]]
        test_df.text_preprocessed.astype(str)
        test_df.oneg.astype(float)
        model = trainer(train_df, output_dir, preproc, args)

    tester(model, task, test_df, preproc, args.runName)


if __name__ == "__main__":
    main()
