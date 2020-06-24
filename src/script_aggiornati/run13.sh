echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 13 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --tagAndUnpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 13 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --removeMention false --normalizeMention false --normalizeUrl true --tagAndUnpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_13_opos.csv --predictNeg res_bert_try_mirko_13_oneg.csv --runName 13 
python sentipolc16_eval.py -r res_bert_to_evaluete_13.csv -t res_13.txt 
echo "FINITO" 
