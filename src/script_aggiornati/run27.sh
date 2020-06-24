echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 27 --task opos --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl true --rawHashtag true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 27 --task oneg --normalizeNoise true --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl true --rawHashtag true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_27_opos.csv --predictNeg res_bert_try_mirko_27_oneg.csv --runName 27 
python sentipolc16_eval.py -r res_bert_to_evaluete_27.csv -t res_27.txt 
echo "FINITO" 
