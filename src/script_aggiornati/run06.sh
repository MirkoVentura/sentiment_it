echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 06 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --unpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 06 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl false --normalizeUrl false --unpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_06_opos.csv --predictNeg res_bert_try_mirko_06_oneg.csv --runName 06 
python sentipolc16_eval.py -r res_bert_to_evaluete_06.csv -t res_06.txt 
echo "FINITO" 
