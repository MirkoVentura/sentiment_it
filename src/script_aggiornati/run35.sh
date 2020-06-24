echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 35 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 35 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_35_opos.csv --predictNeg res_bert_try_mirko_35_oneg.csv --runName 35 
python sentipolc16_eval.py -r res_bert_to_evaluete_35.csv -t res_35.txt 
echo "FINITO" 
