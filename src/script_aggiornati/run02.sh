echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 02 --task opos --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --normalizeMention true --normalizeUrl true --rawHashtag true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 02 --task oneg --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --normalizeMention true --normalizeUrl true --rawHashtag true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_02_opos.csv --predictNeg res_bert_try_mirko_02_oneg.csv --runName 02 
python sentipolc16_eval.py -r res_bert_to_evaluete_02.csv -t res_02.txt 
echo "FINITO" 
