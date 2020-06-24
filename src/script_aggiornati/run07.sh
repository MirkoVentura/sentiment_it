echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 07 --task opos --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python sentiment_ita.py --preproc mirko --runName 07 --task oneg --normalizeNoise true --removeEmoji false --doEmoji false --removeEmoticon true --doEmoticon true --removeMention true --normalizeUrl true --tagAndUnpackHastags true --removePunctuation false --doLower false 
python reducer.py --predictPos res_bert_try_mirko_07_opos.csv --predictNeg res_bert_try_mirko_07_oneg.csv --runName 07 
python sentipolc16_eval.py -r res_bert_to_evaluete_07.csv -t res_07.txt 
echo "FINITO" 
