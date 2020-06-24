echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 25 --task opos --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --rawHashtag true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 25 --task oneg --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --rawHashtag true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_25_opos.csv --predictNeg res_bert_try_mirko_25_oneg.csv --runName 25 
python sentipolc16_eval.py -r res_bert_to_evaluete_25.csv -t res_25.txt 
echo "FINITO" 
