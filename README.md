# LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs
**Unofficial** LemmaTag [1] implementation in TensorFlow 2

## Getting started
Each experiment can be run with the following command:
```
python train.py --scheduler <scheduler> --we-dim <we-dim> --encoder-layers <encoder-layers> --name <project>_<name>
```
For example the original results from [1] can be reproduced using:
```
python train.py --scheduler lemmatag --we-dim 768 --encoder-layers 2 --name tagger_lemmatag
```

## Results
The results including the pre-trained models can be found here: https://app.wandb.ai/kulhanek/lemmatag?workspace=user-kulhanek

| configuration        | lemmatizer dev accuracy | lemmatizer test accuracy | tagger dev accuracy | tagger test accuracy |
|----------------------|------------------------:|-------------------------:|--------------------:|---------------------:|
| baseline             | 99.02                   | 98.75                    | 97.00               | 96.67                |
| original lemmatag [1]| -                       | 98.37                    |                    | 96.90                |

## Tagger-only results
In this experiment the tagger was trained without the lemmatizer. For tagger alone, the training results including the pre-trained models can be found on wandb: https://app.wandb.ai/kulhanek/tagger?workspace=user-kulhanek
| configuration         | scheduler | encoder-layers | we-dim | dev accuracy | test accuracy |
|-----------------------|-----------|----------------|--------|-------------:|--------------:|
| tagger-baseline       | lemmatag  | 2              | 768    | 97.03        | 96.79         |
| tagger-cosine         | cosine    | 2              | 768    | 97.05        | 96.74         |
| tagger-4-layers       | cosine    | 4              | 542    | 97.09        | 96.73         |
| original lemmatag [1] | lemmatag  | 2              | 768    | -            | 96.83         |


## References:
[1] LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs, *Daniel Kondratyuk, Tomáš Gavenčiak, Milan Straka, Jan Hajič* <br/>
https://arxiv.org/abs/1808.03703
