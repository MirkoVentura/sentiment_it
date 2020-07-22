
import argparse
from typing import Tuple
from pathlib import Path

def trainer(file_path: Path,
            filenames: Tuple[str, str, str],
            checkpoint: str,
            stack: str,
            n_epochs: int) -> None:
    """Train sentiment model using Flair NLP library:
    https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md

    To help provide added context, we can stack Glove, Bert or ELMo embeddings along with Flair embeddings.
    """
    # pip install flair allennlp
    from flair.datasets import ClassificationCorpus
    from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from flair.training_utils import EvaluationMetric
    from flair.visual.training_curves import Plotter

    if stack == "glove":
        from flair.embeddings import WordEmbeddings
        stacked_embedding = WordEmbeddings('glove')
    elif stack == "fasttext":
        from flair.embeddings import WordEmbeddings
        stacked_embedding = WordEmbeddings('it')
    elif stack == "elmo":
        from flair.embeddings import ELMoEmbeddings
        stacked_embedding = ELMoEmbeddings('original')
    elif stack == "bert":
        from flair.embeddings import BertEmbeddings
        stacked_embedding = BertEmbeddings('bert-base-uncased')
    elif stack == "bert-multi":
        from flair.embeddings import BertEmbeddings
        stacked_embedding = BertEmbeddings('bert-base-multilingual-uncased')
    elif stack == 'bpe':
        from flair.embeddings import BytePairEmbeddings
        stacked_embedding = BytePairEmbeddings('it') 
    else:
        stacked_embedding = None

    # Define and Load corpus from the provided dataset
    train, dev, test = filenames
    corpus = ClassificationCorpus(
        file_path,
        train_file=train,
        dev_file=dev,
        test_file=test,
    )
    # Create label dictionary from provided labels in data
    label_dict = corpus.make_label_dictionary()

    # Stack Flair string-embeddings with optional embeddings
    word_embeddings = list(filter(None, [
        stacked_embedding,
        FlairEmbeddings('it-forward'),
        FlairEmbeddings('it-backward'),
    ]))
    # Initialize document embedding by passing list of word embeddings
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings,
        hidden_size=256,
        reproject_words=True,
        dropout=0.5,
        reproject_words_dimension=256,
    )
    
    
    #document_embeddings = DocumentPoolEmbeddings([
    #    stacked_embedding,
    #    FlairEmbeddings('it-forward'),
    #    FlairEmbeddings('it-backward')],pooling='mean')
        
    # Define classifier
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        multi_label=True
    )

    if not checkpoint:
        trainer = ModelTrainer(classifier, corpus)
    else:
        # If checkpoint file is defined, resume training
        #checkpoint = classifier.load_checkpoint(Path(checkpoint))
        trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)

    # Begin training (enable checkpointing to continue training at a later time, if desired)
    trainer.train(
        file_path,
        max_epochs=n_epochs,
        checkpoint=True,
    )

    # Plot curves and store weights and losses
    plotter = Plotter()
    plotter.plot_training_curves(file_path + '/loss.tsv')
    plotter.plot_weights(file_path + '/weights.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    filepath ="./data/POLARITY_MULTI"
    filenames = ('train.csv','dev.csv','test.csv')
    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stack', type=str, help="Type of embeddings to stack along with Flair embeddings", default='bpe')
    parser.add_argument('--epochs', type=int, help="Number of epochs", default=500)
    parser.add_argument('--checkpoint', type=str, help="Path of checkpoint file (to restart training)", default=None)

    args = parser.parse_args()

    if args.stack not in ['glove', 'fasttext', 'elmo', 'bert','bert-multi',None,'bpe']:
        parser.error("Please specify only one of glove, elmo or bert for stacked embeddings!")
    
    
    trainer(filepath, filenames, args.checkpoint, stack=args.stack, n_epochs=args.epochs)