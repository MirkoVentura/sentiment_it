echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 23 --task opos --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --normalizeMention true --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 23 --task oneg --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --normalizeMention true --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_23_opos.csv --predictNeg res_bert_try_mirko_23_oneg.csv --runName 23 
python sentipolc16_eval.py -r res_bert_to_evaluete_23.csv -t res_23.txt 
echo "FINITO" 
