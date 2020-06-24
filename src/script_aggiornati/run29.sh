echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 29 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl true --normalizeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 29 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl true --normalizeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_29_opos.csv --predictNeg res_bert_try_mirko_29_oneg.csv --runName 29 
python sentipolc16_eval.py -r res_bert_to_evaluete_29.csv -t res_29.txt 
echo "FINITO" 
