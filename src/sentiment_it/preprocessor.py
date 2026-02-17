"""Reusable text preprocessor with configurable strategies for social media text."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import emoji
import pandas as pd
import wordninja
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

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


class Strategy(Enum):
    RAW = 0
    NORMALIZE = 1
    PACK = 2
    REMOVE = 3
    TRANSLATE = 4
    PACK_TRANSLATE = 5


class TextProcessor:
    """Configurable text preprocessor for social media content.

    Supports Italian (IT) and English (EN) with strategy-based handling
    of emoji, emoticons, URLs, hashtags, mentions, and punctuation.
    """

    def __init__(self, args, lang: str):
        self.others = Strategy(args.others)
        self.emoji_strategy = Strategy(args.emoji)
        self.emoticon = Strategy(args.emoticon)
        self.url = Strategy(args.url)
        self.hashtag = Strategy(args.hashtag)
        self.punctuation = Strategy(args.punctuation)
        self.mention = Strategy(args.mention)
        self.lower = args.lower
        self.lang = lang

        self.ita_moji = pd.read_csv(DATA_DIR / "italianMoji.csv", sep=";")

        if self.lang == "IT":
            self.language_model = wordninja.LanguageModel(str(DATA_DIR / "words.last_all.txt.gz"))
        else:
            self.language_model = None

        self._emoticon_dict = EMOTICON_EN if self.lang == "EN" else EMOTICON_IT

        self.text_processor = TextPreProcessor(
            remove=["email", "percent", "money", "phone", "time", "date", "number"],
            annotate={},
            fix_html=True,
            unpack_hashtags=False,
            tokenizer=SocialTokenizer(lowercase=self.lower).tokenize,
            dicts=[emoticons],
        )

    def do_preprocess(self, tweet: str) -> str:
        """Apply all configured preprocessing strategies to the input tweet."""
        smiley = self._emoticon_dict

        # Emoticon handling
        if self.emoticon == Strategy.REMOVE:
            words = tweet.split()
            tweet = " ".join(" " if w in smiley else w for w in words)
        elif self.emoticon == Strategy.NORMALIZE:
            words = tweet.split()
            tweet = " ".join("emoticon" if w in smiley else w for w in words)
        elif self.emoticon == Strategy.PACK:
            words = tweet.split()
            tweet = " ".join("(emoticon)" if w in smiley else w for w in words)
        elif self.emoticon == Strategy.TRANSLATE:
            words = tweet.split()
            tweet = " ".join(smiley.get(w, w) for w in words)

        # Emoji handling
        if self.emoji_strategy == Strategy.REMOVE:
            tweet = EMOJI_PATTERN.sub(r"", tweet)
        elif self.emoji_strategy == Strategy.NORMALIZE:
            tweet = EMOJI_PATTERN.sub(r" emoji ", tweet)
        elif self.emoji_strategy == Strategy.PACK:
            tweet = EMOJI_PATTERN.sub(r" (emoji) ", tweet)
        elif self.emoji_strategy == Strategy.TRANSLATE:
            if emoji.emoji_count(tweet) != 0:
                if self.lang == "EN":
                    tweet = emoji.demojize(tweet, delimiters=("", ""))
                    tweet = tweet.replace("_", " ")
                elif self.lang == "IT":
                    for emo in emoji.emoji_list(tweet):
                        single_moji = str(emo["emoji"])
                        ita_m = self.ita_moji[self.ita_moji["emoji"] == single_moji]
                        if len(ita_m["text_ita"].values) != 0:
                            tweet = tweet.replace(single_moji, ita_m["text_ita"].values[0])
                        else:
                            tweet = tweet.replace(single_moji, "")
        elif self.emoji_strategy == Strategy.PACK_TRANSLATE:
            if emoji.emoji_count(tweet) != 0:
                if self.lang == "EN":
                    tweet = emoji.demojize(tweet, delimiters=("(", ")"))
                    tweet = tweet.replace("_", " ")
                elif self.lang == "IT":
                    for emo in emoji.emoji_list(tweet):
                        single_moji = str(emo["emoji"])
                        ita_m = self.ita_moji[self.ita_moji["emoji"] == single_moji]
                        if len(ita_m["text_ita"].values) != 0:
                            tweet = tweet.replace(single_moji, f"({ita_m['text_ita'].values[0]})")
                        else:
                            tweet = tweet.replace(single_moji, "")

        # Noise normalization
        if self.others in (Strategy.NORMALIZE, Strategy.PACK):
            tweet = " ".join(self.text_processor.pre_process_doc(tweet))
            if self.lang == "EN":
                tweet = tweet.replace("<percent>", "<percentage>")
            elif self.lang == "IT":
                tweet = tweet.replace("<percent>", "<percentuale>")
                tweet = tweet.replace("<money>", "<soldi>")
                tweet = tweet.replace("<time>", "<tempo>")
                tweet = tweet.replace("<date>", "<data>")
                tweet = tweet.replace("<number>", "<numero>")

            if self.others == Strategy.NORMALIZE:
                tweet = tweet.replace("<", " ").replace(">", " ")
            else:
                tweet = tweet.replace("<", "(").replace(">", ")")

        # Hashtag handling
        elems = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]
        if self.hashtag == Strategy.REMOVE:
            for elem in elems:
                tweet = tweet.replace(f"#{elem}", " ")
        elif self.hashtag == Strategy.TRANSLATE:
            for elem in elems:
                if self.lang == "IT" and self.language_model:
                    translated = " ".join(self.language_model.split(elem))
                else:
                    translated = " ".join(wordninja.split(elem))
                tweet = tweet.replace(f"#{elem}", translated)
        elif self.hashtag == Strategy.PACK_TRANSLATE:
            for elem in elems:
                if self.lang == "IT" and self.language_model:
                    translated = " ".join(self.language_model.split(elem))
                else:
                    translated = " ".join(wordninja.split(elem))
                tweet = tweet.replace(f"#{elem}", f"< {translated} >")
        elif self.hashtag == Strategy.NORMALIZE:
            for elem in elems:
                tweet = tweet.replace(f"#{elem}", "#hashtag")
        elif self.hashtag == Strategy.PACK:
            for elem in elems:
                tweet = tweet.replace(f"#{elem}", "(#hashtag)")

        # URL handling
        if self.url == Strategy.REMOVE:
            tweet = " ".join(re.sub(r"(\w+://\S+)", " ", tweet).split())
        elif self.url == Strategy.NORMALIZE:
            tweet = " ".join(re.sub(r"(\w+://\S+)", " url ", tweet).split())
        elif self.url == Strategy.PACK:
            tweet = " ".join(re.sub(r"(\w+://\S+)", " (url) ", tweet).split())

        # Mention handling
        if self.mention == Strategy.NORMALIZE:
            tweet = " ".join(re.sub(r"(@[A-Za-z0-9]+)", " @user ", tweet).split())
        elif self.mention == Strategy.PACK:
            tweet = " ".join(re.sub(r"(@[A-Za-z0-9]+)", " (@user) ", tweet).split())
        elif self.mention == Strategy.REMOVE:
            tweet = " ".join(re.sub(r"(@[A-Za-z0-9]+)", " ", tweet).split())

        # Punctuation handling
        if self.punctuation == Strategy.REMOVE:
            tweet = " ".join(re.sub(r"[.,!?:;\-=]", " ", tweet).split())

        return tweet.lower() if self.lower else tweet
