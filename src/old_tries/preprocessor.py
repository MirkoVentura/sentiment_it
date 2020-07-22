from bs4 import BeautifulSoup
import re
import itertools
import emoji

def load_dict_emoticon():
    
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

def tweet_cleaning_for_sentiment_analysis(tweet):    
    translator = Translator()

    #Escaping HTML characters
    
    tweet = BeautifulSoup(tweet).get_text()
    
    #Deal with emoticons 
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
   
    #Special case not handled previously.
    tweet = tweet.replace('\x92',"")
    tweet = tweet.replace('\x85',"")

    #Removal of hastags/account
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)", " user ", tweet).split()) #verificare posizione nel testo 
    tweet = ' '.join(re.sub("#", "", tweet).split())

    #|(#[A-Za-z0-9]+)
    #Removal of address
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " url ", tweet).split()) #valutare cancellezione url se a fine frase
    
    #Removal of Punctuation
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\]", " ", tweet).split()) #rimuovere << oppure "" 
    
    
    
    #CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    tweet = tweet.replace("’"," ")
    
    
    # Standardizing words
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    
    number = emoji.emoji_count(tweet)
   
    
    #Deal with emojis
    
    tweet = emoji.demojize(tweet,use_aliases=False,delimiters=("", ""))
    tweet = tweet.replace("_"," ")

    tweet = tweet.replace(":"," ")
    if number != 0:
      tweet = translator.translate(tweet,src='en', dest='it').text

    tweet = ' '.join(tweet.split())
    
    #Lower case
    tweet = tweet.lower()
    return tweet

if __name__ == "__main__":
	data = pd.read_csv('./polita/training_set_sentipolc16.csv')
	test = pd.read_csv('./polita/test_set_sentipolc16_gold2000.csv')


	data.text = data_trimmed.text.astype(str)
	data.text = data_trimmed.text.apply(tweet_cleaning_for_sentiment_analysis)

	print("cleaning TEST****")
	test.text = test_trimmed.text.astype(str)
	test.text = test_trimmed.text.apply(tweet_cleaning_for_sentiment_analysis)


	data.to_csv(r'/data/TASK_BERT/POLARITY_MULTI/train.csv', header=['idtwitter','subj','opos','pred_neg','iro','lpos','lneg','top','text'], index=None, sep='\t', mode='a')
	test.to_csv(r'/data/TASK_BERT/POLARITY_MULTI/test.csv', header=['idtwitter','subj','oneg','pred_neg','iro','lpos','lneg','top','text'], index=None, sep='\t', mode='a')