#!/usr/bin/env bash
set -euo pipefail

echo "Starting English sentiment analysis pipeline..."

RUN_NAME="${1:-01}"

# Train and predict
python -m sentiment_it.sentiment_eng \
    --preproc mirko --runName "$RUN_NAME" \
    --normalizeNoise true --removeEmoji true --doEmoji true \
    --removeEmoticon false --doEmoticon true \
    --removeMention false --normalizeMention false \
    --removeUrl false --normalizeUrl false \
    --rawHashtag true --removePunctuation false --doLower false

echo "Done!"
