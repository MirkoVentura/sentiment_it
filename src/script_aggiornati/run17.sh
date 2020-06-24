echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 17 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --rawHashtag true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 17 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --rawHashtag true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_17_opos.csv --predictNeg res_bert_try_mirko_17_oneg.csv --runName 17 
python sentipolc16_eval.py -r res_bert_to_evaluete_17.csv -t res_17.txt 
echo "FINITO" 
