#!/usr/bin/env python3
import argparse
import sys

from morpho_dataset import MorphoDataset
from data import create_pipelines
from train import Model as LemmatagModel
from model import LemmatagConfig


NUM_TEST_SENTENCES = 10


def output_predictions(predictions, dataset_type, out_path='predictions-{dataset}.txt', test=False):
    out_path = out_path.format(dataset=dataset_type)
    morpho = MorphoDataset(max_sentences=NUM_TEST_SENTENCES if test else None)
    morpho_dataset = getattr(morpho, dataset_type)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, (tags, lemmas) in enumerate(predictions):
            for j in range(len(morpho_dataset.data[morpho_dataset.FORMS].word_strings[i])):
                lemma_string = ''.join(morpho_dataset.data[morpho_dataset.LEMMAS].alphabet[x] for x in lemmas[j])
                print(morpho_dataset.data[morpho_dataset.FORMS].word_strings[i][j],
                      lemma_string,
                      morpho_dataset.data[morpho_dataset.TAGS].words[tags[j]],
                      sep="\t", file=out_file)
            print(file=out_file)


def parse_args():
    argstr = ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='lemmatag-cz', type=str, help="Model to load.")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument('--output', default='predictions-{dataset}.txt')
    args = parser.parse_args([] if "__file__" not in globals() else None)
    return args, argstr


if __name__ == "__main__":
    # Parse arguments
    args, argstr = parse_args()

    # Load the data
    morpho = MorphoDataset()

    # Create the network and train
    config = LemmatagConfig.from_pretrained(args.model)
    model = LemmatagModel.from_pretrained(args.model)
    train_dataset, dev_dataset, test_dataset = create_pipelines(morpho, args, config.tag_configurations)

    output_predictions(model.predict(dev_dataset), 'dev', out_path=args.output)
    output_predictions(model.predict(dev_dataset), 'test', out_path=args.output)
