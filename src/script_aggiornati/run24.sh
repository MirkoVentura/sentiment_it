echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 24 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --normalizeUrl true --removeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 24 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --normalizeUrl true --removeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_24_opos.csv --predictNeg res_bert_try_mirko_24_oneg.csv --runName 24 
python sentipolc16_eval.py -r res_bert_to_evaluete_24.csv -t res_24.txt 
echo "FINITO" 
