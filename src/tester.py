from simpletransformers.classification import ClassificationModel
import csv
import pandas as pd
import argparse
import emoji
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
import os
import json
import wordninja
import numpy as np
from preprocessor import TexProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--others', type=int, help="email end similar", default=0)
    parser.add_argument('--emoji', type=int, help="emoji", default=0)
    parser.add_argument('--emoticon', type=int, help="emoticon", default=0)
    parser.add_argument('--url', type=int, help="url", default=0)
    parser.add_argument('--punctuation', type=int, help="punctuation", default=0)
    parser.add_argument('--mention', type=int, help="mention", default=0)
    parser.add_argument('--hashtag', type=int, help="hashtag", default=0)
    parser.add_argument('--lower', type=bool, help="lower", default=False)
    parser.add_argument('--lang', type=str, help="language ", default='EN')
    
    args = parser.parse_args()
    text_processor_testing = TexProcessor(args,args.lang)

    print(text_processor_testing.do_preprocess('SIDE TO SIDE 😘 @arianagrande #sidetoside '))



