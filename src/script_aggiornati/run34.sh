echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 34 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 34 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_34_opos.csv --predictNeg res_bert_try_mirko_34_oneg.csv --runName 34 
python sentipolc16_eval.py -r res_bert_to_evaluete_34.csv -t res_34.txt 
echo "FINITO" 
