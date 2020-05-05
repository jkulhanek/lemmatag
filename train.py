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
from itertools import chain

from morpho_dataset import MorphoDataset
from data import create_pipelines, collect_tag_configurations
import wandb

NUM_TEST_SENTENCES = 10

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


def Encoder(args, num_words, num_chars, unknown_char): 
    word_ids = tf.keras.layers.Input((None,), dtype=tf.int32, name='word_ids')
    charseqs = tf.keras.layers.Input((None, None,), dtype=tf.int32, name='charseqs') 

    # We will prepare the word embedding and the character-level embedding
    masked_word_ids = MaskedLayer(unknown_char, args.word_dropout)(word_ids)
    we = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(masked_word_ids)
    valid_words = tf.where(word_ids != 0)
    cle = tf.gather_nd(charseqs, valid_words) 
    cle = tf.keras.layers.Embedding(num_chars, args.we_dim // 2, mask_zero=True)(cle)
    cle = tf.keras.layers.Dropout(args.dropout)(cle)
    cle_outputs, cle_state_fwd, cle_state_bwd = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(args.we_dim // 2, return_sequences=True, return_state=True),
    )(cle) 
    cle_states = tf.keras.layers.Concatenate(-1)([cle_state_fwd, cle_state_bwd])
    cle = tf.scatter_nd(valid_words, cle_states, [tf.shape(charseqs)[0], tf.shape(charseqs)[1], cle_states.shape[-1]]) 
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
        x = tf.keras.layers.Add()([x, xres])
    return tf.keras.Model(inputs=[word_ids, charseqs], outputs=[x, cle_states, cle_outputs]) 


class LemmaDecoder(tf.keras.Model):
    def __init__(self, args, num_target_chars, bow, eow):
        super().__init__()
        self.num_target_chars = num_target_chars
        self.rnn_dim = args.we_dim 
        self.target_cle_dim = args.we_dim // 2
        self.eow, self.bow = eow, bow

    def build(self, input_shape):
        self.rnn_cell = tf.keras.layers.LSTMCell(self.rnn_dim) 
        self.attention = tfa.seq2seq.LuongAttention(self.rnn_dim)
        def cell_input_fn(inputs, attention):
            return tf.concat([inputs, attention, self._rnn_states_additional_input], -1)
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.rnn_cell, self.attention,
            cell_input_fn = cell_input_fn, output_attention=False)
        self.temb = tf.keras.layers.Embedding(self.num_target_chars, self.target_cle_dim) 
        self.decoder = tf.keras.layers.Dense(self.num_target_chars)

        training_sampler = tfa.seq2seq.TrainingSampler()
        self.training_decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, training_sampler, self.decoder)

        prediction_sampler = tfa.seq2seq.GreedyEmbeddingSampler(self.temb)
        self.prediction_decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, prediction_sampler, self.decoder)

    def call(self, inputs, training=None, mask=None, target=None):
        """
        input is the tuple (word_ids, we_rnn_outputs, cle_outputs, cle_states, tag_features, target_characters)
        """
        if training is None: training = tf.keras.backend.learning_phase()
        we_rnn_outputs, cle_outputs, cle_states, charseqs_lens, tag_features = inputs 
        if training:
            target_lens = tf.reduce_sum(tf.cast(target != 0, tf.int32), -1)
            target = tf.pad(target, [[0,0],[1,0]], constant_values=self.bow)[:,:-1]

        self.attention.setup_memory(cle_outputs, memory_sequence_length=charseqs_lens)
        self._rnn_states_additional_input = tf.concat([we_rnn_outputs, cle_states, tag_features], -1) 
        words_count = tf.shape(we_rnn_outputs)[0]
        initial_state = self.rnn_cell.get_initial_state(batch_size=words_count, dtype=tf.float32) \
            .clone(cell_state=[we_rnn_outputs, we_rnn_outputs])
        if training:
            target_emb = self.temb(target)
            results, _, result_lengths = self.training_decoder(target_emb,
                initial_state=initial_state, sequence_length=target_lens)

        else: 
            self.prediction_decoder.maximum_iterations = tf.reduce_max(charseqs_lens) + 10
            results, _, result_lengths = self.prediction_decoder(None, start_tokens=tf.tile([self.bow], [words_count]), 
                end_token=self.eow, initial_state=initial_state)
        return tf.nn.softmax(results.rnn_output), results.sample_id, result_lengths


class Model(tf.keras.Model):
    def __init__(self, args, num_words, num_chars, tag_configurations, unknown_char, num_target_chars, bow, eow):
        super().__init__()
        self.encoder = Encoder(args, num_words, num_chars, unknown_char)
        self.tag_decoder = TagDecoder(tag_configurations)
        self.lemmatizer = LemmaDecoder(args, num_target_chars, bow, eow)

    def call(self, inputs, training_targets=None, training=None):
        encoded, cle_states, cle_outputs = self.encoder(inputs, training=training)
        tag_outputs = self.tag_decoder(encoded, mask=encoded._keras_mask, training=training)

        # Prepare target for lemmatizer
        tag_features = tf.stop_gradient(tf.concat(tag_outputs, -1))
        word_ids = inputs[0]
        valid_words = tf.where(word_ids != 0)
        we_rnn_outputs = tf.gather_nd(encoded, valid_words)
        tag_features = tf.gather_nd(tag_features, valid_words)
        cle_seq_lengths = tf.reduce_sum(tf.cast(tf.gather_nd(inputs[1] != 0, valid_words), tf.int32), -1)

        inputs = [we_rnn_outputs, cle_outputs, cle_states, cle_seq_lengths, tag_features]
        lemma_outputs = self.lemmatizer(inputs, target=training_targets,
                mask=encoded._keras_mask, training=training)
        return (tag_outputs, lemma_outputs)


class WordAccuracy(tf.keras.metrics.Mean):
    def __init__(self, name='word_accuracy'):
        super().__init__(name=name)

    def update_state(self, target, prediction):
        correct_values = tf.logical_or(target == prediction, target == 0)
        accuracy = tf.reduce_all(correct_values, -1)
        super().update_state(accuracy)

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
    def __init__(self, args, num_words, tag_configurations, num_chars, unknown_char, prepare_tag_target,
            num_target_chars, eow, **kwargs): 
        self.args = args
        self.eow = eow
        self.model = Model(args, num_words, num_chars, tag_configurations, unknown_char, 
            num_target_chars=num_target_chars, eow=eow, **kwargs) 
        self._learning_schedule = tf.keras.experimental.CosineDecay(args.learning_rate, args.epochs)
        self._optimizer = tfa.optimizers.LazyAdam(args.learning_rate, beta_1=0.9, beta_2=0.99)
        self._metrics = {
            'tagger_accuracy': tf.metrics.CategoricalAccuracy(),
            'lemmatizer_accuracy': WordAccuracy(),
            'tagger_loss': tf.metrics.Mean(),
            'lemmatizer_loss': tf.metrics.Mean(),
            'grad_norm': tf.metrics.Mean(),
            'tagger_loss_direct': tf.metrics.Mean(),
            'tagger_loss_auxiliary': tf.metrics.Mean(),
            'loss': tf.metrics.Mean(),
        }
        self._tag_configurations = tag_configurations 
        self._prepare_tag_target = prepare_tag_target
        self._tag_criterion = self.build_tagger_criterion(args, tag_configurations)
        self._lemma_criterion = self.build_lemmatizer_criterion(args, num_target_chars)
        
    def build_tagger_criterion(self, args, tag_configurations): 
        criterion = [CategoricalCrossentropy(name=f'tagger_loss_{x.name}', label_smoothing=args.label_smoothing) for x in tag_configurations]
        tagger_loss_weights = [x.weight for x in tag_configurations]
        def fn(pred, target, mask):
            losses = [l(gi, pi, sample_weight=mask) for l, gi, pi in zip(criterion, target, pred)]
            tagger_loss = sum(l * w for l, w in zip(losses, tagger_loss_weights))
            self._metrics['tagger_accuracy'].update_state(target[0], pred[0], mask)
            self._metrics['tagger_loss_direct'].update_state(losses[0])
            self._metrics['tagger_loss_auxiliary'].update_state(sum(losses[1:]))
            self._metrics['tagger_loss'].update_state(tagger_loss)
            return tagger_loss
        return fn

    def build_lemmatizer_criterion(self, args, num_target_chars): 
        criterion = CategoricalCrossentropy(name='lemmatizer_loss', label_smoothing=args.label_smoothing)
        def fn(pred, target, training=True):
            pred_value, pred_ids, _ = pred
            mask = tf.cast(target != 0, tf.float32)
            target_onehot = tf.one_hot(target, num_target_chars)
            if not training:
                # align the predictions to target len
                pred_value = pred_value[:,:tf.shape(target)[1],:]
                pred_ids = pred_ids[:,:tf.shape(target)[1]]
                padding = tf.shape(target)[1] - tf.shape(pred_value)[1]
                if padding > 0:
                    pred_value = tf.pad(pred_value, [[0,0],[0, padding],[0,0]],
                        constant_values=1.0/tf.cast(tf.shape(pred_value)[-1], tf.float32))
                    pred_ids = tf.pad(pred_ids, [[0,0],[0, padding]])
            loss = criterion(target_onehot, pred_value, sample_weight=mask)
            self._metrics['lemmatizer_loss'].update_state(loss)
            self._metrics['lemmatizer_accuracy'].update_state(target, pred_ids)
            return loss
        return fn

    @tf.function(experimental_relax_shapes = True)
    def train_on_batch(self, x, y, reset_metrics = True):
        if reset_metrics: self.reset_metrics()
        tagger_target, lemmatizer_target = y
        tagger_mask = tf.cast(x[0] != 0, tf.float32)
        with tf.GradientTape() as tp:
            # Passing lemmatizer target for teacher-forcing
            tagger_pred, lemmatizer_pred = self.model(x, training_targets=lemmatizer_target, training=True)
            tagger_loss = self._tag_criterion(tagger_pred, tagger_target, tagger_mask)
            lemmatizer_loss = self._lemma_criterion(lemmatizer_pred, lemmatizer_target)
            loss = tagger_loss + lemmatizer_loss

        grads = tp.gradient(loss, self.model.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.args.grad_clip)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self._metrics['grad_norm'].update_state(grad_norm) 
        self._metrics['loss'].update_state(loss) 
        result = { k: v.result() for k, v in self._metrics.items() }
        return result


    @tf.function(experimental_relax_shapes = True)
    def test_on_batch(self, x, y, reset_metrics = False):
        if reset_metrics: self.reset_metrics()
        tagger_target, lemmatizer_target = y
        tagger_mask = tf.cast(x[0] != 0, tf.float32)
        with tf.GradientTape() as tp:
            tagger_pred, lemmatizer_pred = self.model(x, training=False)
            tagger_loss = self._tag_criterion(tagger_pred, tagger_target, tagger_mask)
            lemmatizer_loss = self._lemma_criterion(lemmatizer_pred, lemmatizer_target,training=False)
            loss = tagger_loss + lemmatizer_loss

        self._metrics['loss'].update_state(loss) 
        result = { k: v.result() for k, v in self._metrics.items() }
        del result['grad_norm']
        return result


    @tf.function(experimental_relax_shapes = True)
    def predict_on_batch(self, x):
        tagger_pred, lemmatizer_pred = self.model(x, training=False)
        tags = tf.argmax(tagger_pred[0], -1, tf.int32)
        valid_words = tf.where(x[0] != 0)
        lemmas = lemmatizer_pred[1]
        lemmas = tf.scatter_nd(valid_words, lemmas, [tf.shape(tags)[0], tf.shape(tags)[1], tf.shape(lemmas)[-1]])
        return tags, lemmas

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

            print('epoch: {epoch}, loss: {train_loss:.4f}, grad_norm: {train_grad_norm:.4f}, val_loss: {val_loss:.4f}, lemmatizer acc.: {val_lemmatizer_accuracy:.4f}, tagger acc.: {val_tagger_accuracy:.4f}'.format(**logdict))
            if not self.args.test:
                wandb.log(logdict, step=epoch + 1)

            # Save model every fifth epoch
            if (epoch + 1) % 5 == 0:
                self.model.save_weights('model.h5')
                output_predictions(self, self.predict(dev_dataset), 'dev')
                output_predictions(self, self.predict(test_dataset), 'test')
                if not self.args.test:
                    wandb.save('model.h5')
                    wandb.save('predictions-dev.txt')
                    wandb.save('predictions-test.txt')

    def predict(self, dataset):
        def crop_lemma(lemma):
            res = []
            for c in map(int, lemma):
                if c == self.eow: break
                res.append(c)
            return res

        predictions = []
        for x, target in dataset:
            preds = self.predict_on_batch(x)
            for tags, lemmas in zip(*preds):
                lemmas = list(map(crop_lemma, lemmas))
                predictions.append((tags, lemmas))
        return predictions


def output_predictions(model, predictions, dataset_type, out_path='predictions-{dataset}.txt'):
    out_path = out_path.format(dataset=dataset_type) 
    morpho = MorphoDataset(max_sentences=NUM_TEST_SENTENCES if model.args.test else None)
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
        prepare_tag_target=build_prepare_tag_target(morpho.train, tag_configurations),
        num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet),
        bow = morpho.train.data[morpho.train.LEMMAS].alphabet_map['<bow>'],
        eow = morpho.train.data[morpho.train.LEMMAS].alphabet_map['<eow>'])

    data_pipelines = create_pipelines(morpho, args, tag_configurations)

    print(f'running command "{argstr}"')
    network.fit(*data_pipelines)


