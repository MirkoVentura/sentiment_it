echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 05 --task opos --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon false --removeMention true --removeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 05 --task oneg --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon false --removeMention true --removeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_05_opos.csv --predictNeg res_bert_try_mirko_05_oneg.csv --runName 05 
python sentipolc16_eval.py -r res_bert_to_evaluete_05.csv -t res_05.txt 
echo "FINITO" 
