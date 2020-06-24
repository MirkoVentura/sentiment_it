echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 09 --task opos --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 09 --task oneg --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --removeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_09_opos.csv --predictNeg res_bert_try_mirko_09_oneg.csv --runName 09 
python sentipolc16_eval.py -r res_bert_to_evaluete_09.csv -t res_09.txt 
echo "FINITO" 
