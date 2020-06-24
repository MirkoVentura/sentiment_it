echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 33 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 33 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_33_opos.csv --predictNeg res_bert_try_mirko_33_oneg.csv --runName 33 
python sentipolc16_eval.py -r res_bert_to_evaluete_33.csv -t res_33.txt 
echo "FINITO" 
