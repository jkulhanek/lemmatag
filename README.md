# LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs
**Unofficial** LemmaTag [1] implementation in TensorFlow 2

## Implementation status
Currently, only POS tagging is implemented.

## Getting started
Each experiment can be run with the following command:
```
python train.py --scheduler <scheduler> --we-dim <we-dim> --encoder-layers <encoder-layers> --name <project>_<name>
```
For example the original results from [1] can be reproduced using:
```
python train.py --scheduler lemmatag --we-dim 768 --encoder-layers 2 --name tagger_lemmatag
```

## Tagger results
For tagger alone, the training results including the pre-trained models can be found on wandb: https://app.wandb.ai/kulhanek/tagger?workspace=user-kulhanek
| configuration  | scheduler | encoder-layers | we-dim | our accuracy | accuracy reported in [1] |
|----------------|-----------|----------------|--------|--------------|--------------------------|
| lemmatag       | lemmatag  | 2              | 768    |              | 96.83                    |
| cosine         | cosine    | 2              | 768    |              |                          |
| 3-layers       | cosine    | 3              | 628    |              |                          |
| 4-layers       | cosine    | 4              | 542    |              |                          |

## References:
[1] LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs, *Daniel Kondratyuk, Tomáš Gavenčiak, Milan Straka, Jan Hajič* <br/>
https://arxiv.org/abs/1808.03703
