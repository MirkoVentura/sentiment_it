echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 04 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --normalizeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 04 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --normalizeUrl true --normalizeHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_04_opos.csv --predictNeg res_bert_try_mirko_04_oneg.csv --runName 04 
python sentipolc16_eval.py -r res_bert_to_evaluete_04.csv -t res_04.txt 
echo "FINITO" 
