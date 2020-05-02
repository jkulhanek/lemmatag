#!/usr/bin/env python3
import argparse
import datetime
import os, sys
import re

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from collections import namedtuple
from functools import partial

from morpho_dataset import MorphoDataset
from data import create_pipelines, collect_tag_configurations
import wandb

NUM_TEST_SENTENCES = 10

def lemmatag_learning_schedule(lr, epoch):
    return lr * min(1.0, 0.25 ** (1 + ((epoch - 20) // 10)))

LR_SCHEDULES = {
    'lemmatag': lambda lr, epochs: partial(lemmatag_learning_schedule, lr),
    'cosine': tf.keras.experimental.CosineDecay}

class MaskedLayer(tf.keras.layers.Layer):
    def __init__(self, mask_character, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.mask_character = mask_character

    def call(self, x, training=False):
        if training:
            noise = tf.random.uniform(tf.shape(x)) < self.dropout_rate
            noise = tf.logical_and(noise, tf.cast(x, tf.bool)) # apply mask to noise
            return tf.where(noise, self.mask_character, x) 
        else:
            return x

class FirstMaskAdd(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        r = x[0]
        for i in range(1, len(x)):
            r += x[i]
        return r

    def compute_mask(self, inputs, mask=None):
        if mask is None: return None
        return mask[0]

class EncoderCell(tf.keras.layers.Layer):
    def __init__(self, features, dropout_rate): 
        super().__init__()
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(features, return_sequences=True),
            merge_mode='sum')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.features = features

    def call(self, inputs, training=None, mask=None):
        x = self.rnn(inputs, training=training, mask=mask)
        x = self.dropout(x, training=training)
        x += inputs
        return x

class TagDecoder(tf.keras.layers.Layer):
    def __init__(self, tag_configurations):
        super().__init__() 
        self._compute_output_and_mask_jointly = True 
        self.heads = []
        for tag_config in tag_configurations:
            self.heads.append(
                tf.keras.layers.Dense(tag_config.num_values, activation='softmax', name=tag_config.name))

    def call(self, x, training=None, mask=None):
        result = [head(x, training=training) for head in self.heads]
        for r in result: r._keras_mask = mask
        return result

class Encoder(tf.keras.layers.Layer):
    def __init__(self, args, num_words, num_chars, unknown_char = 1):
        super().__init__()
        self.masked_layer = MaskedLayer(unknown_char, args.word_dropout)
        self.we = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)
        self.cle_embedding = tf.keras.layers.Embedding(num_chars, args.we_dim // 2, mask_zero=True)
        self.cle_dropout = tf.keras.layers.Dropout(args.dropout)
        self.cle = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(args.we_dim // 2),
        ) 
        self.joint_dropout = tf.keras.layers.Dropout(args.dropout)
        self.trunk = tf.keras.Sequential([EncoderCell(args.we_dim, args.dropout) for _ in range(args.encoder_layers)])
        self._compute_output_and_mask_jointly = True 

    def call(self, inputs, training=None): 
        word_ids, charseqs = inputs
        masked_word_ids = self.masked_layer(word_ids, training=training)
        we = self.we(masked_word_ids, training=training)
        we_mask = we._keras_mask
        valid_words = tf.where(we_mask)

        cle = tf.gather_nd(charseqs, valid_words) 
        cle = self.cle_embedding(cle, training=training)
        cle_mask = cle._keras_mask
        cle = self.cle(cle, mask=cle_mask, training=training)
        cle = self.cle_dropout(cle, training=training) 
        cle = tf.scatter_nd(valid_words, cle, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], cle.shape[-1]]) 

        joint_embedding = we + cle
        joint_embedding = self.joint_dropout(joint_embedding, training=training)
        embedding = self.trunk(joint_embedding, mask=we_mask, training=training)
        return embedding

def create_model(args, num_words, num_chars, tag_configurations, unknown_char):
    word_ids = tf.keras.layers.Input((None,), dtype=tf.int32, name='word_ids')
    charseqs = tf.keras.layers.Input((None, None,), dtype=tf.int32, name='charseqs') 

    # We will prepare the word embedding and the character-level embedding
    masked_word_ids = MaskedLayer(unknown_char, args.word_dropout)(word_ids)
    we = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(masked_word_ids)
    valid_words = tf.where(word_ids != 0)
    cle = tf.gather_nd(charseqs, valid_words) 
    cle = tf.keras.layers.Embedding(num_chars, args.we_dim // 2, mask_zero=True)(cle)
    cle = tf.keras.layers.Dropout(args.dropout)(cle)
    cle = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(args.we_dim // 2),
    )(cle) 
    cle = tf.scatter_nd(valid_words, cle, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], cle.shape[-1]]) 
    embedded = FirstMaskAdd()([we, cle]) # Used for better performance, targets are masked in the loss

    # We will build the network trunk 
    x = embedded
    x = tf.keras.layers.Dropout(args.dropout)(x)
    for _ in range(args.encoder_layers): 
        xres = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(args.we_dim, return_sequences=True),
            merge_mode='sum'
        )(x)
        xres = tf.keras.layers.Dropout(args.dropout)(xres)
        x, xres = x + xres, None # Destroy xres variable 

    outputs = []
    for tag_config in tag_configurations:
        outputs.append(
            tf.keras.layers.Dense(tag_config.num_values, activation='softmax', name=tag_config.name)(x))
    return tf.keras.Model(inputs=[word_ids, charseqs], outputs=outputs)


def create_model(args, num_words, num_chars, tag_configurations, unknown_char):
    word_ids = tf.keras.layers.Input((None,), dtype=tf.int32, name='word_ids')
    charseqs = tf.keras.layers.Input((None, None,), dtype=tf.int32, name='charseqs') 

    # We will prepare the word embedding and the character-level embedding
    embedded = Encoder(args, num_words, num_chars, unknown_char)([word_ids, charseqs])
    outputs = TagDecoder(tag_configurations)(embedded)
    return tf.keras.Model(inputs=[word_ids, charseqs], outputs=outputs)


# TF FIX - very hacky!
# https://github.com/tensorflow/tensorflow/pull/369901
# TODO: remove after fixed in TF
class CategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self._fix_label_smoothing = label_smoothing

    def call(self, y_true, y_pred): 
        label_smoothing = self._fix_label_smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        return super().call(y_true, y_pred) 


def collect_factors(factor_words): 
    # We will collect all factors from the dataset
    # Since it is not provided in Straka's dump
    factors = []
    for i in range(15):
        unique = list(set([x[i] for x in factor_words if len(x) == 15]))
        unique.sort()
        unique = ['<pad>', '<unk>'] + unique
        factors.append(unique)
    return factors
        

def build_prepare_tag_target(dataset, tag_configurations):
    words = dataset.data[dataset.TAGS].words
    metawords = words[:2]
    def prepare_tag_target(batch):
        word_ids = batch[dataset.TAGS].word_ids
        targets = [np_onehot(word_ids, tag_configurations[0].num_values)]
        batch_words = [words[x] for x in word_ids.reshape((-1,))]
        for i, x in enumerate(tag_configurations[1:]):
            target = [x.lookup.get(w[i], 1) if len(w) > i else x.lookup.get(w, 1) for w in batch_words]
            target = np.array(target, np.int32).reshape(word_ids.shape)
            targets.append(np_onehot(target, x.num_values))
        return targets 
    return prepare_tag_target


class Network:
    def __init__(self, args, num_words, tag_configurations, num_chars, unknown_char, prepare_tag_target): 
        self.args = args
        self.model = create_model(args, num_words, num_chars, tag_configurations, unknown_char) 
        self._learning_schedule = LR_SCHEDULES[args.scheduler](args.learning_rate, args.epochs)
        self._optimizer = tfa.optimizers.LazyAdam(args.learning_rate, beta_1=0.9, beta_2=0.99)
        self._loss = [CategoricalCrossentropy(name=f'loss_{x.name}', label_smoothing=args.label_smoothing) for x in tag_configurations]
        self._loss_weights = [x.weight for x in tag_configurations]
        self._metrics = {
            'tagger_accuracy': tf.metrics.CategoricalAccuracy(),
            'tagger_loss': tf.metrics.Mean(),
            'grad_norm': tf.metrics.Mean(),
            'tagger_loss_direct': tf.metrics.Mean(),
            'tagger_loss_auxiliary': tf.metrics.Mean(),
            'loss': tf.metrics.Mean(),
        }
        self._tag_configurations = tag_configurations 
        self._prepare_tag_target = prepare_tag_target

    @tf.function(experimental_relax_shapes = True)
    def train_on_batch(self, x, y, reset_metrics = True):
        if reset_metrics: self.reset_metrics()
        mask = tf.cast(x[0] != 0, tf.float32)
        with tf.GradientTape() as tp:
            tagger_pred = self.model(x, training=True)
            losses = [l(gi, pi, sample_weight=mask) for l, gi, pi in zip(self._loss, y, tagger_pred)]
            tagger_loss = sum(l * w for l, w in zip(losses, self._loss_weights))
            loss = tagger_loss
        grads = tp.gradient(loss, self.model.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.args.grad_clip)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self._metrics['grad_norm'].update_state(grad_norm) 
        self._metrics['tagger_accuracy'].update_state(y[0], tagger_pred[0], mask)
        self._metrics['tagger_loss'].update_state(tagger_loss)
        self._metrics['tagger_loss_direct'].update_state(losses[0])
        self._metrics['tagger_loss_auxiliary'].update_state(sum(losses[1:]))
        self._metrics['loss'].update_state(loss) 
        result = { k: v.result() for k, v in self._metrics.items() }
        return result


    @tf.function(experimental_relax_shapes = True)
    def test_on_batch(self, x, y, reset_metrics = False):
        if reset_metrics: self.reset_metrics()
        mask = tf.cast(x[0] != 0, tf.float32)
        tagger_pred = self.model(x, training=False)
        losses = [l(gi, pi, sample_weight=mask) for l, gi, pi in zip(self._loss, y, tagger_pred)]
        tagger_loss = sum(l * w for l, w in zip(losses, self._loss_weights))
        loss = tagger_loss

        self._metrics['tagger_accuracy'].update_state(y[0], tagger_pred[0], mask)
        self._metrics['tagger_loss'].update_state(tagger_loss)
        self._metrics['tagger_loss_direct'].update_state(losses[0])
        self._metrics['tagger_loss_auxiliary'].update_state(sum(losses[1:]))
        self._metrics['loss'].update_state(loss) 
        result = { k: v.result() for k, v in self._metrics.items() }
        del result['grad_norm']
        return result


    @tf.function(experimental_relax_shapes = True)
    def predict_on_batch(self, x):
        return self.model(x, training=False)

    def reset_metrics(self):
        for m in self._metrics.values(): m.reset_states()


    def fit(self, train_dataset, dev_dataset, test_dataset):
        if not self.args.test:
            wandb.init(project=self.args.project, name=self.args.name)
            wandb.config.update(self.args)

        for epoch in range(self.args.epochs):
            logdict = dict(epoch=epoch + 1)
            lr = self._learning_schedule(epoch)
            K.set_value(self._optimizer.lr, K.get_value(lr))
            logdict['learning_rate'] = lr

            for x, target in train_dataset:
                metrics = self.train_on_batch(x, target, reset_metrics=False) 
            self.reset_metrics()
            logdict.update({f'train_{n}': v.numpy() for n,v in metrics.items()})

            for x, target in dev_dataset:
                metrics = self.test_on_batch(x, target, reset_metrics=False)
            self.reset_metrics()
            logdict.update({f'val_{n}': v.numpy() for n,v in metrics.items()})

            print(f'epoch: {logdict["epoch"]}, loss: {logdict["train_tagger_loss"]:.4f}, grad_norm: {logdict["train_grad_norm"]:.4f}, val_loss: {logdict["val_tagger_loss"]:.4f}, accuracy: {logdict["val_tagger_accuracy"]:.4f}')
            if not self.args.test:
                wandb.log(logdict, step=epoch + 1)

            # Save model every fifth epoch
            if (epoch + 1) % 5 == 0:
                self.model.save_weights('model.h5')
                output_predictions(self, dev_dataset, 'dev')
                output_predictions(self, test_dataset, 'test')
                if not self.args.test:
                    wandb.save('model.h5')
                    wandb.save('tagger-dev.txt')
                    wandb.save('tagger-test.txt')

    def predict(self, dataset):
        predictions = []
        for x, target in dataset:
            tagger_preds = self.predict_on_batch(x)
            for pred in tagger_preds[0]:
                sentence = np.argmax(pred, -1)
                predictions.append(sentence)
        return predictions


def output_predictions(model, dataset, dataset_type, out_path='tagger-{dataset}.txt'):
    out_path = out_path.format(dataset=dataset_type) 
    morpho = MorphoDataset(max_sentences=NUM_TEST_SENTENCES if model.args.test else None)
    morpho_dataset = getattr(morpho, dataset_type)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(dataset)):
            for j in range(len(morpho_dataset.data[morpho_dataset.FORMS].word_strings[i])):
                print(morpho_dataset.data[morpho_dataset.FORMS].word_strings[i][j],
                      morpho_dataset.data[morpho_dataset.LEMMAS].word_strings[i][j],
                      morpho_dataset.data[morpho_dataset.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)




def parse_args(): 
    argstr = ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--word-dropout", default=0.25, type=float)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--grad-clip", default=3.0, type=float)
    # IMPORTANT: LemmaTag paper uses lemmatag scheduler, but default is cosine (works better)
    parser.add_argument("--scheduler", default='cosine', choices=['lemmatag', 'cosine'], help='The lemmatag paper uses lemmatag learning rate scheduler, however cosine decay works better, therefore it is left here as default')
    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--encoder-layers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--we-dim", default=768, type=int, help="Word embedding dimension.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--name", default=(os.environ['JOB'] if 'JOB' in os.environ else 'tagger_baseline'))
    args = parser.parse_args([] if "__file__" not in globals() else None)

    assert '_' in args.name 
    args.project, args.name = args.name[:args.name.index('_')], args.name[args.name.index('_') + 1:]
    return args, argstr


if __name__ == "__main__":
    # Parse arguments
    args, argstr = parse_args()

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load the data
    morpho = MorphoDataset(max_sentences=NUM_TEST_SENTENCES if args.test else None)
    tag_configurations = list(collect_tag_configurations(args, morpho.train))

    # Create the network and train
    unknown_char = morpho.train.data[morpho.train.FORMS].words.index('<unk>')
    network = Network(args,
        num_words=len(morpho.train.data[morpho.train.FORMS].words),
        tag_configurations = tag_configurations,
        num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
        unknown_char=unknown_char,
        prepare_tag_target=build_prepare_tag_target(morpho.train, tag_configurations))

    data_pipelines = create_pipelines(morpho, args, tag_configurations)

    print(f'running command "{argstr}"')
    network.fit(*data_pipelines)

