echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 18 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 18 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_18_opos.csv --predictNeg res_bert_try_mirko_18_oneg.csv --runName 18 
python sentipolc16_eval.py -r res_bert_to_evaluete_18.csv -t res_18.txt 
echo "FINITO" 
