echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 14 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 14 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_14_opos.csv --predictNeg res_bert_try_mirko_14_oneg.csv --runName 14 
python sentipolc16_eval.py -r res_bert_to_evaluete_14.csv -t res_14.txt 
echo "FINITO" 
