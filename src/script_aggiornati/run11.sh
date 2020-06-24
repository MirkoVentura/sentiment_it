echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 11 --task opos --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --unpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 11 --task oneg --normalizeNoise false --removeEmoji false --doEmoji true --removeEmoticon true --doEmoticon true --removeMention true --removeUrl false --normalizeUrl false --unpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_11_opos.csv --predictNeg res_bert_try_mirko_11_oneg.csv --runName 11 
python sentipolc16_eval.py -r res_bert_to_evaluete_11.csv -t res_11.txt 
echo "FINITO" 
