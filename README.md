# sentiment_it

Sentiment analysis for Italian and English social media text using BERT-based models.

This project implements sentiment classification for tweets using fine-tuned BERT models, with extensive text preprocessing strategies for handling emoji, emoticons, hashtags, URLs, and mentions.

## Features

- **Italian sentiment analysis** using [dbmdz/bert-base-italian-xxl-uncased](https://huggingface.co/dbmdz/bert-base-italian-xxl-uncased) and [AlBERTo](https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0)
- **English sentiment analysis** using `bert-base-cased`
- **Configurable text preprocessing** with strategy-based pipeline (translate, normalize, pack, remove)
- **SENTIPOLC 2016** evaluation support (opos/oneg tasks)
- **SemEval 2017 Task 4A** evaluation support

## Project Structure

```
sentiment_it/
├── pyproject.toml              # Project metadata, dependencies, tool config
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff, trailing whitespace)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── src/sentiment_it/           # Python package
│   ├── __init__.py
│   ├── sentiment_ita.py        # Italian sentiment pipeline
│   ├── sentiment_eng.py        # English sentiment pipeline
│   ├── preprocessor.py         # Text preprocessing with Strategy pattern
│   ├── reducer.py              # Combine opos/oneg predictions
│   ├── tester.py               # Preprocessor testing utility
│   └── sentipolc16_eval.py     # SENTIPOLC 2016 evaluation script
├── data/                       # Datasets and configuration
│   ├── conf.json               # BERT training hyperparameters
│   ├── italianMoji.csv         # Italian emoji translations
│   └── ...                     # Training/test datasets
├── scripts/                    # Execution scripts
│   ├── run_ita.sh
│   └── run_eng.sh
└── tests/                      # Test suite
```

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package with dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Usage

### Italian Sentiment Analysis

```bash
python -m sentiment_it.sentiment_ita \
    --preproc mirko \
    --task opos \
    --runName 01 \
    --doEmoji true \
    --doEmoticon true \
    --normalizeNoise true
```

### English Sentiment Analysis

```bash
python -m sentiment_it.sentiment_eng \
    --preproc mirko \
    --runName 01 \
    --doEmoji true \
    --doEmoticon true
```

### Preprocessing Strategies

The `TextProcessor` class in `preprocessor.py` supports these strategies for each text element:

| Strategy | Value | Description |
|----------|-------|-------------|
| RAW | 0 | Keep as-is |
| NORMALIZE | 1 | Replace with generic tag |
| PACK | 2 | Wrap in parentheses |
| REMOVE | 3 | Delete entirely |
| TRANSLATE | 4 | Convert to text representation |
| PACK_TRANSLATE | 5 | Translate and wrap in parentheses |

### Evaluation

```bash
python -m sentiment_it.sentipolc16_eval \
    -r results.csv \
    -g data/sentipolc16_gold2000.csv \
    -t results_report
```

## Development

```bash
# Run linter
ruff check src/

# Run formatter
ruff format src/

# Run tests
pytest
```

## References

- [SENTIPOLC 2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/) - Sentiment Polarity Classification shared task
- [SemEval 2017 Task 4](https://alt.qcri.org/semeval2017/task4/) - Sentiment Analysis in Twitter
