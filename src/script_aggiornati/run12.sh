echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 12 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --normalizeUrl true --unpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 12 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --normalizeUrl true --unpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_12_opos.csv --predictNeg res_bert_try_mirko_12_oneg.csv --runName 12 
python sentipolc16_eval.py -r res_bert_to_evaluete_12.csv -t res_12.txt 
echo "FINITO" 
