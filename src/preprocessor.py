from enum import Enum
import emoji
import wordninja
import re
import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import json

class Strategy(Enum):
    RAW = 0
    NORMALIZE = 1
    PACK = 2
    REMOVE = 3
    TRASLATE = 4
    PACK_TRASLATE = 5
    

class TexProcessor(object):

    def __init__ (self, args,lang) :
        self.others = Strategy(args.others) # valori validi 0,1,2 //ANDATA
        self.emoji = Strategy(args.emoji) #0,1 emoji ,2 (emoji) ,3, 4 ,5 (traduzione) //
        self.emoticon = Strategy(args.emoticon) #0,1 emoticon ,2 (emoticon) ,3, 4 ,5 (traduzione)
        self.url = Strategy(args.url) # 0,1,2,3
        self.hashtag = Strategy(args.hashtag) # 0,1 = #hashtag,2 ,3 (#hashtag),4,5
        self.punctuation = Strategy(args.punctuation) #Valori validi 0,3
        self.mention = Strategy(args.mention) #0,1,2,3
        self.lower = args.lower #true o false
        self.lang = lang # EN o IT 
        self.ita_moji = pd.read_csv('./data/italianMoji.csv',sep=';')
        if self.lang == 'IT':
            self.lm = wordninja.LanguageModel('./data/words.last_all.txt.gz')
        else:
            self.lm = None
        self.text_processor = TextPreProcessor (
            remove=[ 
                'email' , #raw o nomralize.
                'percent', #raw o nomralize: EN: percentage, IT: percentuale.
                'money', # raw o nomralize: EN: money, IT: soldi. verificare se becca le valute
                'phone', # raw o nomralize: EN: phone, IT: telefono
                'time', # raw o nomralize: EN time, It: ore 
                'date', # raw o nomralize EN date, It data
                'number' #raw o nomralize En number, it numero.
            ] ,
            annotate={} ,
            fix_html=True ,
            unpack_hashtags=False ,
            tokenizer=SocialTokenizer(lowercase=self.lower).tokenize,
            dicts = [ emoticons ])
    


    def load_dict_emoticon(self):
        if self.lang == 'EN':
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
        else:
            return {
        ":‑)":"felice",
        ": ‑)":"felice",
        ": ‑ )":"felice",
        ":-]":"felice",
        ": - ]":"felice",
        ": -]":"felice",
        ":-3":"felice",
        ": - 3":"felice",
        ": -3":"felice",
        ":->":"felice",
        ": - >":"felice",
        ": ->":"felice",
        "8-)":"felice",
        "8 -)":"felice",
        "8 - )":"felice",
        ":-}":"felice",
        ": - }":"felice",
        ": -}":"felice",
        ":)":"felice",
        ": )":"felice",
        ":]":"felice",
        ": ]":"felice",
        ":3":"felice",
        ": 3":"felice",
        ":>":"felice",
        ": >":"felice",
        "8)":"felice",
        "8 )":"felice",
        ":}":"felice",
        ": }":"felice",
        ":o)":"felice",
        ":o )":"felice",
        ": o )":"felice",
        ":c)":"felice",
        ": c )":"felice",
        ":c )":"felice",
        ":^)":"felice",
        ": ^ )":"felice",
        ": ^)":"felice",
        "=]":"felice",
        "= ]":"felice",
        "=)":"felice",
        "= )":"felice",
        ":-))":"felice",
        ": - ) )":"felice",
        ":- ) )":"felice",
        ":- ))":"felice",
        ": -))":"felice",
        ":‑D":"felice",
        ": ‑ D":"felice",
        ": ‑D":"felice",
        "8‑D":"felice",
        "8 ‑D":"felice",
        "8 ‑ D":"felice",
        "x‑D":"felice",
        "x ‑ D":"felice",
        "x ‑D":"felice",
        "X‑D":"felice",
        "X ‑ D":"felice",
        "X ‑D":"felice",
        ":D":"felice",
        ": D":"felice",
        "8D":"felice",
        "8 D":"felice",
        "xD":"felice",
        "x D":"felice",
        "XD":"felice",
        "X D":"felice",
        ":‑(":"triste",
        ": ‑(":"triste",
        ": ‑ (":"triste",
        ":‑c":"triste",
        ": ‑c":"triste",
        ":‑<":"triste",
        ": ‑ <":"triste",
        ":‑[":"triste",
        ": ‑ [":"triste",
        ":(":"triste",
        ": (":"triste",
        ":c":"triste",
        ": c":"triste",
        ":<":"triste",
        ": <":"triste",
        ":[":"triste",
        ": [":"triste",
        ":-||":"triste",
        ": - | |":"triste",
        ": - ||":"triste",
        ": -||":"triste",
        ": -| |":"triste",
        ">:[":"triste",
        ">: [":"triste",
        "> : [":"triste",
        ":{":"triste",
        ": {":"triste",
        ":@":"triste",
        ": @":"triste",
        ">:(":"triste",
        "> : (":"triste",
        ":'‑(":"triste",
        ": '‑(":"triste",
        ": ' ‑(":"triste",
        ": ' ‑ (":"triste",
        ":'(":"triste",
        ": ' (":"triste",
        ": '(":"triste",
        ":‑P":"scherzoso",
        ": ‑P":"scherzoso",
        ": ‑ P":"scherzoso",
        "X‑P":"scherzoso",
        "X ‑ P":"scherzoso",
        "X ‑P":"scherzoso",
        "x‑p":"scherzoso",
        "x ‑p":"scherzoso",
        ":‑p":"scherzoso",
        ": ‑p":"scherzoso",
        ": ‑ p":"scherzoso",
        ":‑Þ":"scherzoso",
        ": ‑ Þ":"scherzoso",
        ":‑þ":"scherzoso",
        ": ‑þ":"scherzoso",
        ":‑b":"scherzoso",
        ": ‑ b":"scherzoso",
        ": ‑b":"scherzoso",
        ":P":"scherzoso",
        ": P":"scherzoso",
        "XP":"scherzoso",
        "X P":"scherzoso",
        "xp":"scherzoso",
        "x p":"scherzoso",
        ":p":"scherzoso",
        ": p":"scherzoso",
        ":Þ":"scherzoso",
        ": Þ":"scherzoso",
        ":þ":"scherzoso",
        ": þ":"scherzoso",
        ":b":"scherzoso",
        ": b":"scherzoso",
        "<3":"amore",
        "< 3":"amore",
        ":*":"amore",
        ": *":"amore"
        }
    

    
    def do_preprocess(self,tweet) :
        #Gestione Emoticon.
        SMILEY = self.load_dict_emoticon() 
        if self.emoticon == Strategy.REMOVE :
            words = tweet.split()
            reformed = [" " if word in SMILEY else word for word in words]
            tweet = " ".join(reformed)
        if self.emoticon == Strategy.NORMALIZE :
            words = tweet.split()
            reformed = ["emoticon" if word in SMILEY else word for word in words]
            tweet = " ".join(reformed)
        if self.emoticon == Strategy.PACK:
            words = tweet.split()
            reformed = ["(emoticon)" if word in SMILEY else word for word in words]
            tweet = " ".join(reformed)
        if self.emoticon == Strategy.TRASLATE:
            words = tweet.split()
            reformed = [SMILEY[word] if word in SMILEY else word for word in words]
            tweet = " ".join(reformed)


        emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  #emoticons 
                u"\U0001F300-\U0001F5FF"     #symbols & pictographs
                u"\U0001F680-\U0001F6FF"     #transport & map symbols
                u"\U0001F1E0-\U0001F1FF"     #flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)

        #Gestione Emoji 
        if self.emoji == Strategy.REMOVE :
            tweet = emoji_pattern.sub(r'', tweet)
        if self.emoji == Strategy.NORMALIZE:
            tweet = emoji_pattern.sub(r' emoji ', tweet)
        if self.emoji == Strategy.PACK :
            tweet = emoji_pattern.sub(r' (emoji) ', tweet)
        if self.emoji == Strategy.TRASLATE:
            number = emoji.emoji_count(tweet)
            if number != 0 :
                if self.lang == 'EN':
                    tweet = emoji.demojize(tweet,delimiters=("", ""))
                    tweet = tweet.replace("_"," ")
                elif self.lang == 'IT':
                    emos = emoji.emoji_lis(tweet)
                    for emo in emos:
                        singleMoji = str(emo['emoji'])
                        ita_M = self.ita_moji[self.ita_moji['emoji'] == singleMoji]
                        if(len(ita_M['text_ita'].values) != 0):
                            significato = ita_M['text_ita'].values[0]
                            tweet = tweet.replace(singleMoji,significato)
                        else:
                            tweet = tweet.replace(singleMoji,'')
        if self.emoji == Strategy.PACK_TRASLATE:
            number = emoji.emoji_count(tweet)
            if number != 0 :
                if self.lang == 'EN':
                    tweet = emoji.demojize(tweet,delimiters=("(", ")"))
                    tweet = tweet.replace("_"," ")
                elif self.lang == 'IT':
                    emos = emoji.emoji_lis(tweet)
                    for emo in emos:
                        singleMoji = str(emo['emoji'])
                        ita_M = self.ita_moji[self.ita_moji['emoji'] == singleMoji]
                        if(len(ita_M['text_ita'].values) != 0):
                            significato = ita_M['text_ita'].values[0]
                            tweet = tweet.replace(singleMoji,'('+significato+')')
                        else:
                            tweet = tweet.replace(singleMoji,'')
        #Gestione other.
        if self.others == Strategy.NORMALIZE:
            tweet = str(" ".join(self.text_processor.pre_process_doc(tweet)))
            if self.lang == 'EN':
                tweet = tweet.replace('<percent>','<percentage>') 
            if self.lang == 'IT':
                tweet = tweet.replace('<percent>','<percentuale>') 
                tweet = tweet.replace('<money>','<soldi>') 
                tweet = tweet.replace('<time>','<tempo>') 
                tweet = tweet.replace('<date>','<data>') 
                tweet = tweet.replace('<number>','<numero>')                 
            tweet = tweet.replace("<"," ")
            tweet = tweet.replace(">"," ")
        if self.others == Strategy.PACK:
            tweet = str(" ".join(self.text_processor.pre_process_doc(tweet)))
            if self.lang == 'EN':
                tweet = tweet.replace('<percent>','<percentage>') 
            if self.lang == 'IT':
                tweet = tweet.replace('<percent>','<percentuale>') 
                tweet = tweet.replace('<money>','<soldi>') 
                tweet = tweet.replace('<time>','<tempo>') 
                tweet = tweet.replace('<date>','<data>') 
                tweet = tweet.replace('<number>','<numero>')   
            tweet = tweet.replace("<","(")
            tweet = tweet.replace(">","(")
        elems = [tag.strip("#") for tag in tweet.split() if tag.startswith("#")]

        #Hashtag
        if self.hashtag == Strategy.REMOVE:
            for elem in elems:
                tweet = tweet.replace("#"+elem," ")
        if self.hashtag == Strategy.TRASLATE:
            for elem in elems:
                if self.lang == 'IT':
                    traslate = ' '.join(self.lm.split(elem))
                    tweet = tweet.replace("#"+elem,traslate)
                if self.lang == 'EN':
                    traslate = ' '.join(wordninja.split(elem))
                    tweet = tweet.replace("#"+elem,traslate)
        if self.hashtag == Strategy.PACK_TRASLATE:
            for elem in elems:
                if self.lang == 'IT':
                    traslate = ' '.join(self.lm.split(elem))
                    tweet = tweet.replace("#"+elem,'< '+traslate+' >')
                if self.lang == 'EN':
                    traslate = ' '.join(wordninja.split(elem))
                    tweet = tweet.replace("#"+elem,'< '+traslate+' >')
        if self.hashtag == Strategy.NORMALIZE:
            for elem in elems:
                tweet = tweet.replace("#"+elem,"#hashtag")
        if self.hashtag == Strategy.PACK:
            for elem in elems:
                tweet = tweet.replace("#"+elem,"(#hashtag)")
        #URLs
        if self.url == Strategy.REMOVE:
            tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
        if self.url == Strategy.NORMALIZE:
            tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " url ", tweet).split())
        if self.url == Strategy.PACK:
            tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " (url) ", tweet).split())
        #Mentions
        if self.mention == Strategy.NORMALIZE:
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " @user ", tweet).split())
        if self.mention == Strategy.PACK:
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " (@user) ", tweet).split())
        if self.mention == Strategy.REMOVE:
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", tweet).split())
        if self.punctuation == Strategy.REMOVE:
            tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
        if self.lower == True:
            return tweet.lower()
        else:
            return tweet
            


