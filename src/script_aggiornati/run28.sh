echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 28 --task opos --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --normalizeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 28 --task oneg --normalizeNoise false --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --normalizeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_28_opos.csv --predictNeg res_bert_try_mirko_28_oneg.csv --runName 28 
python sentipolc16_eval.py -r res_bert_to_evaluete_28.csv -t res_28.txt 
echo "FINITO" 
