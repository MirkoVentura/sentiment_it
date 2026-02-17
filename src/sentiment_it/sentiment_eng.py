"""English sentiment analysis using BERT models (SemEval 2017 tasks)."""

from __future__ import annotations

import argparse
import csv
import html
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

EMOTICON_EN = {
    ":‑)": "happy", ": ‑)": "happy", ": ‑ )": "happy",
    ":-]": "happy", ": - ]": "happy", ": -]": "happy",
    ":-3": "happy", ": - 3": "happy", ": -3": "happy",
    ":->": "happy", ": - >": "happy", ": ->": "happy",
    "8-)": "happy", "8 -)": "happy", "8 - )": "happy",
    ":-}": "happy", ": - }": "happy", ": -}": "happy",
    ":)": "happy", ": )": "happy",
    ":]": "happy", ": ]": "happy",
    ":3": "happy", ": 3": "happy",
    ":>": "happy", ": >": "happy",
    "8)": "happy", "8 )": "happy",
    ":}": "happy", ": }": "happy",
    ":o)": "happy", ":o )": "happy", ": o )": "happy",
    ":c)": "happy", ": c )": "happy", ":c )": "happy",
    ":^)": "happy", ": ^ )": "happy", ": ^)": "happy",
    "=]": "happy", "= ]": "happy",
    "=)": "happy", "= )": "happy",
    ":-))": "happy", ": - ) )": "happy",
    ":- ) )": "happy", ":- ))": "happy", ": -))": "happy",
    ":‑D": "happy", ": ‑ D": "happy", ": ‑D": "happy",
    "8‑D": "happy", "8 ‑D": "happy", "8 ‑ D": "happy",
    "x‑D": "happy", "x ‑ D": "happy", "x ‑D": "happy",
    "X‑D": "happy", "X ‑ D": "happy", "X ‑D": "happy",
    ":D": "happy", ": D": "happy",
    "8D": "happy", "8 D": "happy",
    "xD": "happy", "x D": "happy",
    "XD": "happy", "X D": "happy",
    ":‑(": "sad", ": ‑(": "sad", ": ‑ (": "sad",
    ":‑c": "sad", ": ‑c": "sad",
    ":‑<": "sad", ": ‑ <": "sad",
    ":‑[": "sad", ": ‑ [": "sad",
    ":(": "sad", ": (": "sad",
    ":c": "sad", ": c": "sad",
    ":<": "sad", ": <": "sad",
    ":[": "sad", ": [": "sad",
    ":-||": "sad", ": - | |": "sad",
    ": - ||": "sad", ": -||": "sad", ": -| |": "sad",
    ">:[": "sad", ">: [": "sad", "> : [": "sad",
    ":{": "sad", ": {": "sad",
    ":@": "sad", ": @": "sad",
    ">:(": "sad", "> : (": "sad",
    ":'‑(": "sad", ": '‑(": "sad",
    ": ' ‑(": "sad", ": ' ‑ (": "sad",
    ":'(": "sad", ": ' (": "sad", ": '(": "sad",
    ":‑P": "playful", ": ‑P": "playful", ": ‑ P": "playful",
    "X‑P": "playful", "X ‑ P": "playful", "X ‑P": "playful",
    "x‑p": "playful", "x ‑p": "playful",
    ":‑p": "playful", ": ‑p": "playful", ": ‑ p": "playful",
    ":‑Þ": "playful", ": ‑ Þ": "playful",
    ":‑þ": "playful", ": ‑þ": "playful",
    ":‑b": "playful", ": ‑ b": "playful", ": ‑b": "playful",
    ":P": "playful", ": P": "playful",
    "XP": "playful", "X P": "playful",
    "xp": "playful", "x p": "playful",
    ":p": "playful", ": p": "playful",
    ":Þ": "playful", ": Þ": "playful",
    ":þ": "playful", ": þ": "playful",
    ":b": "playful", ": b": "playful",
    "<3": "love", "< 3": "love",
    ":*": "love", ": *": "love",
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


def add_pred(row, model):
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

    model = ClassificationModel("bert", "bert-base-cased", args=model_param, num_labels=3)
    model.train_model(train_df)
    return model


def tester(model, test_df, preproc, run_name):
    filename = f"./res_bert_en_{preproc}_{run_name}"
    test_df["pred"] = test_df.apply(lambda row: add_pred(row, model), axis=1)
    test_df.pred.replace(1, "positive", inplace=True)
    test_df.pred.replace(0, "neutral", inplace=True)
    test_df.pred.replace(-1, "negative", inplace=True)
    to_save_df = test_df[["id", "pred"]]
    to_save_df.to_csv(filename, index=None, sep="\t", mode="a", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


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
    if isinstance(tweet, float):
        return tweet
    tweet = html.unescape(tweet)

    if args.normalizeNoise:
        tweet = " ".join(text_processor.pre_process_doc(tweet))

    if args.doEmoticon:
        words = tweet.split()
        tweet = " ".join(EMOTICON_EN.get(w, w) for w in words)

    if args.removeEmoticon:
        words = tweet.split()
        tweet = " ".join(" " if w in EMOTICON_EN else w for w in words)

    if args.normalizeEmoticon:
        words = tweet.split()
        tweet = " ".join("<emoticon>" if w in EMOTICON_EN else w for w in words)

    if args.doEmoji:
        if emoji.emoji_count(tweet) != 0:
            tweet = emoji.demojize(tweet, delimiters=("", ""))
            tweet = tweet.replace("_", " ")

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
            translated = " ".join(wordninja.split(elem))
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
    parser = argparse.ArgumentParser(description="English sentiment analysis with BERT")
    parser.add_argument("--odir", type=str, help="Output directory", default="./")
    parser.add_argument("--trainSet", type=str, help="Training CSV path", default=str(DATA_DIR / "combined.tsv"))
    parser.add_argument("--testSet", type=str, help="Test CSV path", default=str(DATA_DIR / "SemEval2017-task4-test.subtask-A.csv"))
    parser.add_argument("--preproc", type=str, help="Preprocessor: mirko or raw", default="mirko")
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

    parser.add_argument("--runName", type=str, help="Run name identifier", default="000")
    parser.add_argument("--modelConf", type=str, help="Model config path", default="conf.json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_train = pd.read_csv(args.trainSet, sep="\t", encoding="utf-8", engine="c")
    data_test = pd.read_csv(args.testSet, sep=";", encoding="utf-8")

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
        data_train.text.astype(str)
        data_test.text.astype(str)
        data_train["text_preprocessed"] = data_train.apply(lambda row: mirko_preprocessing(row, args, text_processor), axis=1)
        data_test["text_preprocessed"] = data_test.apply(lambda row: mirko_preprocessing(row, args, text_processor), axis=1)
    elif preproc == "raw":
        data_train["text_preprocessed"] = data_train["text"]
        data_test["text_preprocessed"] = data_test["text"]

    pd.set_option("display.max_colwidth", 800)
    print("***** TRAIN HEAD ***")
    print(data_train.head())
    print("***** TEST HEAD ***")
    print(data_test.head())

    output_dir = args.odir + f"predictor_{preproc}_{args.runName}"
    train_df = data_train[["text_preprocessed", "sentiment"]]
    train_df.text_preprocessed.astype(str)
    train_df.sentiment.replace("positive", 1, inplace=True)
    train_df.sentiment.replace("neutral", 0, inplace=True)
    train_df.sentiment.replace("negative", -1, inplace=True)
    train_df = train_df[train_df.sentiment != "sentiment"]
    train_df.sentiment.astype(float)

    test_df = data_test[["id", "text_preprocessed", "sentiment"]]
    test_df = test_df[test_df.sentiment != "sentiment"]
    test_df.text_preprocessed.astype(str)

    model = trainer(train_df, output_dir, preproc, args)
    model.eval_model(test_df)
    tester(model, test_df, preproc, args.runName)


if __name__ == "__main__":
    main()
