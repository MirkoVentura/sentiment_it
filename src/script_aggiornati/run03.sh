echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 03 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --removeUrl true --rawHashtag true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 03 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --removeUrl true --rawHashtag true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_03_opos.csv --predictNeg res_bert_try_mirko_03_oneg.csv --runName 03 
python sentipolc16_eval.py -r res_bert_to_evaluete_03.csv -t res_03.txt 
echo "FINITO" 
