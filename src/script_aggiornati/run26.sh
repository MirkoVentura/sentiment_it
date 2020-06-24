echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 26 --task opos --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --rawHashtag true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 26 --task oneg --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --rawHashtag true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_26_opos.csv --predictNeg res_bert_try_mirko_26_oneg.csv --runName 26 
python sentipolc16_eval.py -r res_bert_to_evaluete_26.csv -t res_26.txt 
echo "FINITO" 
