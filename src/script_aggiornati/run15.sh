echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 15 --task opos --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 15 --task oneg --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl true --removeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_15_opos.csv --predictNeg res_bert_try_mirko_15_oneg.csv --runName 15 
python sentipolc16_eval.py -r res_bert_to_evaluete_15.csv -t res_15.txt 
echo "FINITO" 
