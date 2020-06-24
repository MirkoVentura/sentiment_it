echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 22 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon false --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 22 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon false --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_22_opos.csv --predictNeg res_bert_try_mirko_22_oneg.csv --runName 22 
python sentipolc16_eval.py -r res_bert_to_evaluete_22.csv -t res_22.txt 
echo "FINITO" 
