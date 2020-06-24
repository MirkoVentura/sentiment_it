echo "PARTITO!!!" 
. ~/.bashrc 
conda activate poweraiFG 
cd /home/users/gargiulo/PycharmProjects/sentiment_ITA 
python sentiment_ita.py --preproc mirko --runName 32 --task opos --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl false --normalizeUrl false --tagAndUnpackHastags true --removePunctuation true --doLower false 
python sentiment_ita.py --preproc mirko --runName 32 --task oneg --normalizeNoise true --removeEmoji true --doEmoji true --removeEmoticon true --doEmoticon true --normalizeMention true --removeUrl false --normalizeUrl false --tagAndUnpackHastags true --removePunctuation true --doLower false 
python reducer.py --predictPos res_bert_try_mirko_32_opos.csv --predictNeg res_bert_try_mirko_32_oneg.csv --runName 32 
python sentipolc16_eval.py -r res_bert_to_evaluete_32.csv -t res_32.txt 
echo "FINITO" 
