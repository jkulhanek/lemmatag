import numpy as np
import tensorflow as tf

class TagConfiguration:
    def __init__(self, name, values, weight):
        self.name = name
        self.num_values = len(values)
        self.weight = weight
        self.alphabet = values
        self.lookup = {v:i for i,v in enumerate(values)}


def np_onehot(x, depth, dtype=np.float32):
    y = np.zeros(tuple(x.shape) + (depth,), dtype=dtype)
    np.put_along_axis(y, np.expand_dims(x, -1), 1., -1)
    return y

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
        

def collect_tag_configurations(args, dataset):
    class _iter:
        def __iter__(self):
            factors = collect_factors(dataset.data[dataset.TAGS].words)
            yield TagConfiguration('all', dataset.data[dataset.TAGS].words, 1.0)
            for i, f in enumerate(factors):
                yield TagConfiguration(f'f{i + 1}', f, 0.1)
    return _iter()

def create_pipelines(morpho_dataset, args, tag_configurations = None):
    if tag_configurations is None:
        tag_configurations = collect_tag_configurations(args, morpho_dataset.train)
    add_tag_target_fn = add_tag_target(morpho_dataset.train, list(tag_configurations))
    # padded_batch_shape = [[None,], [None, None,],] + \
    #    [[None, x.num_values] for x in tag_configurations]
    padded_batch_shape = dict(words=[None,], charseqs=[None,None], tags=[None,], lemmas=[None, None])

    def unflatten_training_data(x):
        return ([x[0], x[1]], [x[i + 2] for i,_ in enumerate(tag_configurations)])

    prepare_training_data_kwargs = dict(
        unknown_char=morpho_dataset.train.data[morpho_dataset.train.LEMMAS].alphabet_map['<unk>'],
        eow=morpho_dataset.train.data[morpho_dataset.train.LEMMAS].alphabet_map['<eow>'])
    train = morpho_to_tf_dataset(morpho_dataset.train) \
        .cache() \
        .shuffle(3000) \
        .padded_batch(args.batch_size, padded_batch_shape) \
        .map(add_tag_target_fn) \
        .map(prepare_training_data(training=True, **prepare_training_data_kwargs)) \
        .prefetch(4)

    dev = morpho_to_tf_dataset(morpho_dataset.dev) \
        .cache() \
        .padded_batch(args.batch_size, padded_batch_shape) \
        .map(add_tag_target_fn) \
        .map(prepare_training_data(training=False, **prepare_training_data_kwargs)) \
        .prefetch(4)

    test = morpho_to_tf_dataset(morpho_dataset.test) \
        .padded_batch(args.batch_size, padded_batch_shape) \
        .map(add_tag_target_fn) \
        .map(prepare_training_data(training=False, **prepare_training_data_kwargs)) \
        .prefetch(4)
    return train, dev, test

def add_eow(charseqs, eow):
    charseqs = tf.pad(charseqs, [[0,0],[0,1]])
    indices = tf.reduce_sum(tf.cast(charseqs != 0, tf.int32), -1)
    updates = tf.tile([eow], [tf.shape(indices)[0]])
    indices = tf.stack([tf.range(tf.shape(indices)[0]), indices], -1)
    return tf.tensor_scatter_nd_update(charseqs, indices, updates) 


def add_tag_target(dataset, tag_configurations):
    words = dataset.data[dataset.TAGS].words
    metawords = words[:2]
    def prepare_tag_target(word_ids):
        targets = [np_onehot(word_ids, tag_configurations[0].num_values)]
        batch_words = [words[x] for x in word_ids.reshape((-1,))]
        for i, x in enumerate(tag_configurations[1:]):
            target = [x.lookup.get(w[i], 1) if len(w) > i else x.lookup.get(w, 1) for w in batch_words]
            target = np.array(target, np.int32).reshape(word_ids.shape)
            targets.append(np_onehot(target, x.num_values))
        return targets

    @tf.function
    def fn(x):
        x = dict(**x)
        x['tags'] = tf.numpy_function(prepare_tag_target, [x['tags']], [tf.float32 for _ in tag_configurations])
        for i, (t, tag_config) in enumerate(zip(x['tags'], tag_configurations)):
            x['tags'][i] = tf.ensure_shape(t, (None, None, tag_config.num_values)) 
        x['tags'] = tuple(x['tags'])
        return x
    return fn


def morpho_to_tf_dataset(morpho):
    def to_rectangle(charseqs):
        maxlen = max(map(len, charseqs))
        result = np.zeros((len(charseqs), maxlen), np.int32)
        for j, charseq in enumerate(charseqs):
            result[j, :len(charseq)] = charseq
        return result

    def _generate():
        for i in range(morpho.size()):
            yield dict(words = morpho.data[morpho.FORMS].word_ids[i],
                charseqs = to_rectangle(morpho.data[morpho.FORMS].charseqs[i]),
                tags = morpho.data[morpho.TAGS].word_ids[i],
                lemmas = to_rectangle(morpho.data[morpho.LEMMAS].charseqs[i]))

    dataset = tf.data.Dataset.from_generator(_generate, dict(words=tf.int32, charseqs=tf.int32, tags=tf.int32, lemmas=tf.int32))
    return dataset

def prepare_training_data(unknown_char, eow, training=False):
    @tf.function
    def fn(x):
        inputs = (x['words'], x['charseqs'])
        if training: inputs = inputs + (x['lemmas'],)

        # Prepare lemma target
        valid_words = tf.where(inputs[0] != 0)
        lemma_target = tf.gather_nd(x['lemmas'], valid_words)
        lemma_target = add_eow(lemma_target, eow)

        return inputs, (x['tags'], lemma_target,)
    return fn
