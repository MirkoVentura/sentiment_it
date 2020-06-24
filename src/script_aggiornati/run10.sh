echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 10 --task opos --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeUrl false --normalizeUrl false --normalizeHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 10 --task oneg --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeUrl false --normalizeUrl false --normalizeHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_10_opos.csv --predictNeg res_bert_try_mirko_10_oneg.csv --runName 10 
python sentipolc16_eval.py -r res_bert_to_evaluete_10.csv -t res_10.txt 
echo "FINITO" 
