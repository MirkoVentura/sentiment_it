echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 31 --task opos --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl false --normalizeUrl false --tagAndUnpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 31 --task oneg --normalizeNoise false --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl false --normalizeUrl false --tagAndUnpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_31_opos.csv --predictNeg res_bert_try_mirko_31_oneg.csv --runName 31 
python sentipolc16_eval.py -r res_bert_to_evaluete_31.csv -t res_31.txt 
echo "FINITO" 
