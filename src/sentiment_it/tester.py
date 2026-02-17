"""Manual testing script for the text preprocessor."""

from __future__ import annotations

import argparse

from sentiment_it.preprocessor import TextProcessor


def main():
    parser = argparse.ArgumentParser(description="Test text preprocessor")
    parser.add_argument("--others", type=int, help="Others strategy (0-5)", default=0)
    parser.add_argument("--emoji", type=int, help="Emoji strategy (0-5)", default=0)
    parser.add_argument("--emoticon", type=int, help="Emoticon strategy (0-5)", default=0)
    parser.add_argument("--url", type=int, help="URL strategy (0-3)", default=0)
    parser.add_argument("--punctuation", type=int, help="Punctuation strategy (0,3)", default=0)
    parser.add_argument("--mention", type=int, help="Mention strategy (0-3)", default=0)
    parser.add_argument("--hashtag", type=int, help="Hashtag strategy (0-5)", default=0)
    parser.add_argument("--lower", type=bool, help="Lowercase output", default=False)
    parser.add_argument("--lang", type=str, help="Language: EN or IT", default="EN")

    args = parser.parse_args()
    text_processor = TextProcessor(args, args.lang)

    print(text_processor.do_preprocess("SIDE TO SIDE \U0001f618 @arianagrande #sidetoside "))


if __name__ == "__main__":
    main()
