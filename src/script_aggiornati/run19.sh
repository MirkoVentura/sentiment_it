echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 19 --task opos --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention true --normalizeUrl true --unpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 19 --task oneg --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon false --doEmoticon true --removeMention true --normalizeUrl true --unpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_19_opos.csv --predictNeg res_bert_try_mirko_19_oneg.csv --runName 19 
python sentipolc16_eval.py -r res_bert_to_evaluete_19.csv -t res_19.txt 
echo "FINITO" 
