echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 01 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --rawHashtag true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 01 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --rawHashtag true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_01_opos.csv --predictNeg res_bert_try_mirko_01_oneg.csv --runName 01 
python sentipolc16_eval.py -r res_bert_to_evaluete_01.csv -t res_01.txt 
echo "FINITO" 
