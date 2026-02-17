#!/usr/bin/env bash
set -euo pipefail

echo "Starting Italian sentiment analysis pipeline..."

RUN_NAME="${1:-01}"

# Train and predict positive sentiment
python -m sentiment_it.sentiment_ita \
    --preproc mirko --runName "$RUN_NAME" --task opos \
    --normalizeNoise true --removeEmoji true --doEmoji true \
    --removeEmoticon false --doEmoticon true \
    --removeMention false --normalizeMention false \
    --removeUrl false --normalizeUrl false \
    --rawHashtag true --removePunctuation false --doLower false

# Train and predict negative sentiment
python -m sentiment_it.sentiment_ita \
    --preproc mirko --runName "$RUN_NAME" --task oneg \
    --normalizeNoise true --removeEmoji true --doEmoji true \
    --removeEmoticon false --doEmoticon true \
    --removeMention false --normalizeMention false \
    --removeUrl false --normalizeUrl false \
    --rawHashtag true --removePunctuation false --doLower false

# Combine predictions
python -m sentiment_it.reducer \
    --predictPos "res_bert_try_mirko_${RUN_NAME}_opos.csv" \
    --predictNeg "res_bert_try_mirko_${RUN_NAME}_oneg.csv" \
    --runName "$RUN_NAME"

# Evaluate
python -m sentiment_it.sentipolc16_eval \
    -r "res_bert_to_evaluete_${RUN_NAME}.csv" \
    -t "res_${RUN_NAME}"

echo "Done!"
