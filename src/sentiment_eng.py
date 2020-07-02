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
import html
# commento di Rosario

#lm = wordninja.LanguageModel()

def add_pred_pos (row,model,task):
  predictions, raw_outputs = model.predict([row.text_preprocessed])
  if(task == 'opos'):
    return predictions[0]
  else:
    return predictions[0]

def trainer(train_df,OUTPUT_DIR,preproc,args):
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, args.modelConf)
    with open(abs_file_path) as f:
        model_param = json.loads(f.read())
    model_param['output_dir'] = OUTPUT_DIR
    print(model_param)
    model_name = 'bert-large-uncased'
    if preproc == 'alberto':
        model_name = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'
    model = ClassificationModel('bert', model_name, args=model_param,num_labels=3)
    model.train_model(train_df);
    return model

def tester(model,task,test_df,preproc,runName):
    filename = './res_bert_try_'+preproc+'_'+runName
    if task == 'opos':
        filename += '_opos.csv'
        test_df['pred_pos'] = test_df.apply(lambda row: add_pred_pos(row,model,task), axis=1)
    else:
        filename += '_oneg.csv'
        test_df['pred_neg'] = test_df.apply(lambda row: add_pred_pos(row,model,task), axis=1)

    test_df.to_csv(filename,index=None, sep='|', mode='a', quotechar='"',quoting=csv.QUOTE_NONNUMERIC)



def load_dict_emoticon():
    
    return {
        ":‑)":"happy",
        ": ‑)":"happy",
        ": ‑ )":"happy",
        ":-]":"happy",
        ": - ]":"happy",
        ": -]":"happy",
        ":-3":"happy",
        ": - 3":"happy",
        ": -3":"happy",
        ":->":"happy",
        ": - >":"happy",
        ": ->":"happy",
        "8-)":"happy",
        "8 -)":"happy",
        "8 - )":"happy",
        ":-}":"happy",
        ": - }":"happy",
        ": -}":"happy",
        ":)":"happy",
        ": )":"happy",
        ":]":"happy",
        ": ]":"happy",
        ":3":"happy",
        ": 3":"happy",
        ":>":"happy",
        ": >":"happy",
        "8)":"happy",
        "8 )":"happy",
        ":}":"happy",
        ": }":"happy",
        ":o)":"happy",
        ":o )":"happy",
        ": o )":"happy",
        ":c)":"happy",
        ": c )":"happy",
        ":c )":"happy",
        ":^)":"happy",
        ": ^ )":"happy",
        ": ^)":"happy",
        "=]":"happy",
        "= ]":"happy",
        "=)":"happy",
        "= )":"happy",
        ":-))":"happy",
        ": - ) )":"happy",
        ":- ) )":"happy",
        ":- ))":"happy",
        ": -))":"happy",
        ":‑D":"happy",
        ": ‑ D":"happy",
        ": ‑D":"happy",
        "8‑D":"happy",
        "8 ‑D":"happy",
        "8 ‑ D":"happy",
        "x‑D":"happy",
        "x ‑ D":"happy",
        "x ‑D":"happy",
        "X‑D":"happy",
        "X ‑ D":"happy",
        "X ‑D":"happy",
        ":D":"happy",
        ": D":"happy",
        "8D":"happy",
        "8 D":"happy",
        "xD":"happy",
        "x D":"happy",
        "XD":"happy",
        "X D":"happy",
        ":‑(":"sad",
        ": ‑(":"sad",
        ": ‑ (":"sad",
        ":‑c":"sad",
        ": ‑c":"sad",
        ":‑<":"sad",
        ": ‑ <":"sad",
        ":‑[":"sad",
        ": ‑ [":"sad",
        ":(":"sad",
        ": (":"sad",
        ":c":"sad",
        ": c":"sad",
        ":<":"sad",
        ": <":"sad",
        ":[":"sad",
        ": [":"sad",
        ":-||":"sad",
        ": - | |":"sad",
        ": - ||":"sad",
        ": -||":"sad",
        ": -| |":"sad",
        ">:[":"sad",
        ">: [":"sad",
        "> : [":"sad",
        ":{":"sad",
        ": {":"sad",
        ":@":"sad",
        ": @":"sad",
        ">:(":"sad",
        "> : (":"sad",
        ":'‑(":"sad",
        ": '‑(":"sad",
        ": ' ‑(":"sad",
        ": ' ‑ (":"sad",
        ":'(":"sad",
        ": ' (":"sad",
        ": '(":"sad",
        ":‑P":"playful",
        ": ‑P":"playful",
        ": ‑ P":"playful",
        "X‑P":"playful",
        "X ‑ P":"playful",
        "X ‑P":"playful",
        "x‑p":"playful",
        "x ‑p":"playful",
        ":‑p":"playful",
        ": ‑p":"playful",
        ": ‑ p":"playful",
        ":‑Þ":"playful",
        ": ‑ Þ":"playful",
        ":‑þ":"playful",
        ": ‑þ":"playful",
        ":‑b":"playful",
        ": ‑ b":"playful",
        ": ‑b":"playful",
        ":P":"playful",
        ": P":"playful",
        "XP":"playful",
        "X P":"playful",
        "xp":"playful",
        "x p":"playful",
        ":p":"playful",
        ": p":"playful",
        ":Þ":"playful",
        ": Þ":"playful",
        ":þ":"playful",
        ": þ":"playful",
        ":b":"playful",
        ": b":"playful",
        "<3":"love",
        "< 3":"love",
        ":*":"love",
        ": *":"love"
        }


def albertoPreprocessing(row,text_processor) :     
    s = row.text
    s = s.lower()
    s = str(" ".join(text_processor.pre_process_doc(s)))
    s = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00011000-\U0011ffff]", ' ', s)
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r'(\w)\1{2,}',r'\1\1', s)
    s = re.sub ( r'^\s' , '' , s )
    s = re.sub ( r'\s$' , '' , s )
    return s
  
def mirkoPreprocessing(row,args,text_processor):
  SMILEY = load_dict_emoticon()  
  tweet = row.text
  if(type(tweet) == float) :
  	print('---------- float? ------------')
  	print(tweet)
  	print(row)
  	return tweet
  tweet = html.unescape(tweet)


  if args.normalizeNoise:
    tweet = str(" ".join(text_processor.pre_process_doc(tweet)))

  if args.doEmoticon == True:
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)

  if args.removeEmoticon == True:
    words = tweet.split()
    reformed = [" " if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)

  if args.normalizeEmoticon == True:
    words = tweet.split()
    reformed = ["<emoticon>" if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
    
  if args.doEmoji == True:
    number = emoji.emoji_count(tweet)
    if number != 0:
    	tweet = emoji.demojize(tweet,delimiters=("", ""))
    	tweet = tweet.replace("_"," ")
  if args.removeEmoji == True:
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons 
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
  if args.normalizeEmoji == True:
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons 
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r' <emoji> ', tweet)
  if args.removeMention == True:
      tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", tweet).split())
  if args.normalizeMention == True:
      tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " @user ", tweet).split())
  if args.removeUrl == True:
  	tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
  if args.normalizeUrl == True:
      tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " url ", tweet).split())
  if args.rawHashtag == False:
    elems = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]
    for elem in elems:
        traslate = ' '.join(wordninja.split(elem))
        if args.unpackHastags == True:
            tweet = tweet.replace("#"+elem,traslate)
        if args.tagAndUnpackHastags == True:
            tweet = tweet.replace("#"+elem,"<"+ traslate +">")
        if args.removeHastags == True:
            tweet = tweet.replace("#"+elem," ")
        if args.normalizeHastags == True:
            tweet = tweet.replace("#"+elem,"#hashtag")
  if args.removePunctuation:
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
  tweet = re.sub(r"\s+", ' ', tweet)   
  if args.doLower:
  	return tweet.lower()
  else:
    return tweet




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="Type task for bulid model", default='opos')
    parser.add_argument('--odir', type=str, help="output dir", default='./')
    parser.add_argument('--trainSet', type=str, help="Path of csv for train ", default='./data/SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
    parser.add_argument('--testSet', type=str, help="Path of csv for train ", default='./data/SemEval2017-task4-test.subtask-A.csv')
    parser.add_argument('--preproc', type=str, help="Preprocessor approch (possibility alberto,mirko,raw) ", default='mirko')
    parser.add_argument('--doShuffle', type=bool, help="do emoji translation", default=True)

    parser.add_argument('--doEmoji', type=bool, help="do emoji translation", default=True)
    parser.add_argument('--removeEmoji', type=bool, help="do emoji truncation", default=False)
    parser.add_argument('--normalizeEmoji', type=bool, help="do emoji normalization", default=False)
    parser.add_argument('--doEmoticon', type=bool, help="do emoticon translation", default=True)
    parser.add_argument('--removeEmoticon', type=bool, help="do emoticon truncation", default=False)
    parser.add_argument('--normalizeEmoticon', type=bool, help="do emoticon truncation", default=False)
    parser.add_argument('--normalizeMention', type=bool, help="replace User Mention from tweet with @user", default=False)
    parser.add_argument('--removeMention', type=bool, help="remove User Mention", default=False)
    parser.add_argument('--normalizeUrl', type=bool, help="replace url with the word url", default=False)
    parser.add_argument('--removeUrl', type=bool, help="remove url", default=False)
    parser.add_argument('--unpackHastags', type=bool, help="unpack hashtag for mirko preproc", default=False)
    parser.add_argument('--tagAndUnpackHastags', type=bool, help="tag and unpack hashtag for mirko preproc", default=False)
    parser.add_argument('--normalizeHastags', type=bool, help="replace hashtag for mirko preproc with word #hashtag", default=False)
    parser.add_argument('--removeHastags', type=bool, help="remove hashtag for mirko preproc", default=False)
    parser.add_argument('--removePunctuation', type=bool, help="unpack hashtag for mirko preproc", default=False)
    parser.add_argument('--normalizeNoise', type=bool, help="unpack hashtag for mirko preproc", default=False)
    parser.add_argument('--rawHashtag', type=bool, help="unpack hashtag for mirko preproc", default=False)
    parser.add_argument('--doLower', type=bool, help="unpack hashtag for mirko preproc", default=True)


    parser.add_argument('--runName', type=str, help="name for run ", default='000')
    parser.add_argument('--modelConf', type=str, help="path for model hyperparameter configuration ", default='data/conf.json')

    args = parser.parse_args()
    print(args)
    DATA_PATH_ITA = args.trainSet
    DATA_PATH_TEST_ITA = args.testSet
    OUTPUT_DIR = args.odir
    task =  args.task
    preproc = args.preproc
    data_train = pd.read_csv(DATA_PATH_ITA,sep=';',encoding='utf-8',engine='c')
    data_test = pd.read_csv(DATA_PATH_TEST_ITA,sep=";",encoding='utf_8')

    print('***** TRAIN HEAD ***')
    print(data_train.head())    

    if args.doShuffle == True:
    	data_train = data_train.reindex(np.random.permutation(data_train.index))


    if preproc == 'mirko':
        text_processor = TextPreProcessor (
            remove=[ 'email', 'percent', 'money', 'phone', 'time', 'date', 'number'] ,
            annotate={} ,
            fix_html=True ,
            unpack_hashtags=False ,  
            tokenizer=SocialTokenizer(lowercase=args.doLower).tokenize,
            dicts = [ emoticons ])
        data_train.text.astype(str)
        data_test.text.astype(str)
        data_train['text_preprocessed'] = data_train.apply(lambda row: mirkoPreprocessing(row,args,text_processor), axis=1)
        data_test['text_preprocessed'] = data_test.apply(lambda row: mirkoPreprocessing(row,args,text_processor), axis=1)
    if preproc == 'alberto':

        text_processor = TextPreProcessor (
            remove=[ 'url' , 'email', 'user', 'percent', 'money', 'phone', 'time', 'date', 'number'] ,
            annotate={"hashtag"} ,
            fix_html=True ,
            unpack_hashtags=True ,  
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts = [ emoticons ])
        data_train['text_preprocessed'] = data_train.apply(lambda row: albertoPreprocessing(row,text_processor), axis=1)
        data_test['text_preprocessed'] = data_test.apply(lambda row: albertoPreprocessing(row,text_processor), axis=1)
    if preproc == 'raw':
        data_train['text_preprocessed'] = data_train['text']
        data_test['text_preprocessed'] = data_test['text']

    pd.set_option('display.max_colwidth', 800)
    print('***** TRAIN HEAD ***')
    print(data_train.head())    
    print('***** TEST HEAD ***')
    print(data_test.head())
    
    if(task == 'opos'):
        OUTPUT_DIR += 'predictor_all_pos_'+preproc+'_'+args.runName
        train_df = data_train[['text_preprocessed','sentiment']]
        train_df.text_preprocessed.astype(str)
        train_df.sentiment.replace('positive',1,inplace=True)
        train_df.sentiment.replace('neutral',0,inplace=True)
        train_df.sentiment.replace('negative',-1,inplace=True)
        train_df = train_df[[train_df.sentiment != 'sentiment']]
        train_df.sentiment.astype(float)
        test_df = data_test[['text_preprocessed','sentiment']]
        test_df = test_df[[test_df.sentiment != 'sentiment']]
        test_df.text_preprocessed.astype(str)  

        #test_df.opos.astype(float)
        model = trainer(train_df,OUTPUT_DIR,preproc,args)
    if(task == 'oneg'):
        OUTPUT_DIR += 'predictor_all_neg_'+preproc+'_'+args.runName
        train_df = data_train[['text_preprocessed','sentiment']]
        train_df.text_preprocessed.astype(str)
        train_df.oneg.astype(float)
        test_df = data_test[['text_preprocessed','sentiment']]  
        test_df.text_preprocessed.astype(str)  
        test_df.oneg.astype(float)
        model = trainer(train_df,OUTPUT_DIR,preproc,args)
    model.eval_model(test_df)
#    tester(model,task,test_df,preproc,args.runName)