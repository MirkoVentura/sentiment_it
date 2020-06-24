echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 16 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 16 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --removeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_16_opos.csv --predictNeg res_bert_try_mirko_16_oneg.csv --runName 16 
python sentipolc16_eval.py -r res_bert_to_evaluete_16.csv -t res_16.txt 
echo "FINITO" 
