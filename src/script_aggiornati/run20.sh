echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 20 --task opos --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --removeUrl true --unpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 20 --task oneg --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon false --doEmoticon true --normalizeMention true --removeUrl true --unpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_20_opos.csv --predictNeg res_bert_try_mirko_20_oneg.csv --runName 20 
python sentipolc16_eval.py -r res_bert_to_evaluete_20.csv -t res_20.txt 
echo "FINITO" 
