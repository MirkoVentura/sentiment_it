echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 21 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --removeMention true --removeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 21 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon false --doEmoticon true --removeMention true --removeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_21_opos.csv --predictNeg res_bert_try_mirko_21_oneg.csv --runName 21 
python sentipolc16_eval.py -r res_bert_to_evaluete_21.csv -t res_21.txt 
echo "FINITO" 
