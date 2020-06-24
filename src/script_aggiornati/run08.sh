echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 08 --task opos --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --removeMention false --normalizeMention false --removeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 08 --task oneg --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon false --removeMention false --normalizeMention false --removeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_08_opos.csv --predictNeg res_bert_try_mirko_08_oneg.csv --runName 08 
python sentipolc16_eval.py -r res_bert_to_evaluete_08.csv -t res_08.txt 
echo "FINITO" 
