echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 30 --task opos --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl true --unpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 30 --task oneg --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --removeMention false --normalizeMention false --removeUrl true --unpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_30_opos.csv --predictNeg res_bert_try_mirko_30_oneg.csv --runName 30 
python sentipolc16_eval.py -r res_bert_to_evaluete_30.csv -t res_30.txt 
echo "FINITO" 
