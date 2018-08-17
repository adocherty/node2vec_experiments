#
#  Code modified from tensorflow models tutorial:
#  https://www.tensorflow.org/tutorials/representation/word2vec
#

import os
import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
import pickle
import tensorflow as tf

from sklearn import (exceptions)

import warnings

# Ignore metric warnings from scikit-learn. These often occur with
# multi-label prediction when there are few examples of some classes.
warnings.simplefilter("ignore", exceptions.UndefinedMetricWarning)

times = {}


@contextmanager
def timeit(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    times[name] = elapsedTime + times.get(name, 0)


class PregeneratedDataset:
    def __init__(self, data_filename, n_nodes, delimiter=" ", force_offset=0, splits=[0.5, 0.5]):
        self.splits = splits
        self.n_splits = len(splits)
        self.data_index = [0] * self.n_splits
        self.split_sizes = []
        self.split_data = []
        self.vocab_size = n_nodes
        self.existing_vocab = []
        # self.labels = None
        self.affected_nodes = None
        self.unigrams = None
        self.affected_nodes = []
        self.force_offset = force_offset
        self.delimiter = delimiter

        self.build_dataset(data_filename)

    def set_node_degrees(self, degree_file):
        degrees = pd.read_csv(degree_file, delimiter=self.delimiter,
                              dtype='int32', header=None).values

        self.existing_vocab = degrees[:, 0] + self.force_offset
        self.unigrams = np.zeros((self.vocab_size,), dtype=np.int32)
        self.unigrams[self.existing_vocab] = degrees[:, 1]
        self.unigrams = self.unigrams.tolist()

    def epoch_done(self, batch_size=0, split=0):
        return self.data_index[split] + batch_size > self.split_sizes[split]

    def reset_index(self, split=0):
        self.data_index[split] = 0

    def build_freeze_indices(self):
        # Freeze existing vertex-ids excluding (affected vertices + non-existing vertex-ids)
        all_ids = np.arange(self.vocab_size)
        unfreeze_ids = np.delete(all_ids, self.existing_vocab)
        unfreeze_ids = np.append(unfreeze_ids, self.affected_nodes)
        print(
            "all_ids: {}\texisting_vocabs: {}\tafs: {}".format(len(all_ids),
                                                               len(self.existing_vocab),
                                                               len(self.affected_nodes)))
        return np.delete(all_ids, unfreeze_ids)

    def build_dataset(self, data_filename):
        """Process raw inputs into a dataset."""

        # Load all data
        print("Loading target-context pairs from {}".format(data_filename))
        self.data = pd.read_csv(data_filename,
                                delimiter=self.delimiter,
                                dtype='int32',
                                header=None,
                                engine='python').values

        # Force an adjustment to the node indices
        self.data += self.force_offset

        n_total = len(self.data)
        self.split_sizes = [int(n_total * split) for split in self.splits]
        self.split_offset = [0] + self.split_sizes[:-1]
        self.data_index = [0] * self.n_splits

    def set_affected_nodes(self, affected_vertices_file):
        """Set the affected vertices"""
        self.affected_nodes = pd.read_csv(affected_vertices_file,
                                          delimiter=self.delimiter,
                                          dtype='int32',
                                          header=None,
                                          engine='python').values
        self.affected_nodes += self.force_offset

    def generate_batch(self, batch_size, split=0):
        """
        Generate data as (target_word, context_word) pairs.
        """
        data_size = self.split_sizes[split]
        data_offset = self.data_index[split] + self.split_offset[split]

        # Variable batch size - ensure model can handle this
        batch_size = min(batch_size, data_size - self.data_index[split])

        batch = np.empty(batch_size, dtype=np.int32)
        labels = np.empty(batch_size, dtype=np.int32)

        batch[:] = self.data[data_offset: data_offset + batch_size, 0]
        labels[:] = self.data[data_offset: data_offset + batch_size, 1]

        self.data_index[split] += batch_size

        return batch, labels


class W2V_Sampled:
    def __init__(self, embedding_size, vocabulary_size,
                 batch_size=100,
                 val_batch_size=None,
                 save_path=None,
                 learning_rate=0.2,
                 neg_samples=64,
                 lr_decay=0.1):

        self.vocabulary_size = vocabulary_size  #
        self.batch_size = batch_size  #
        self.val_batch_size = val_batch_size

        self.embedding_size = embedding_size  # Embeddings
        self.analogy_k = 4
        self.num_sampled = neg_samples  # Number of negative examples to sample.
        self.learning_rate = learning_rate  # Initial learning rate
        self.lr_decay = lr_decay * 1e-6  # LR exponential decay
        self.save_path = save_path  # Where to store TF output

        self._model_variables = set()

    def save_embeddings(self, epoch, embeddings):
        with open(os.path.join(self.save_path, "embeddings{}.pkl".format(epoch)), "wb") as f:
            pickle.dump(embeddings, f)
        if epoch == 10:
            print(embeddings)

    def load_embeddings(self):
        pass

    def optimize_graph(self, loss, freeze_vars=None, train_vars=None):
        """Build the graph to optimize the loss function."""

        # Global step
        self.global_step = tf.Variable(0, name="global_step")

        lr = self.learning_rate * tf.exp(
            -tf.cast(self.global_step, tf.float32) * self.lr_decay)

        # Instead of running optimizer.minimize directly, call compute gradients
        # and process returned gradients
        optimizer = tf.train.AdagradOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss)

        # Remove frozen indices from gradients
        processes_grads_and_vars = []
        for (g, v) in grads_and_vars:
            if freeze_vars and (v in freeze_vars):
                freeze_indices = freeze_vars[v]

                # Remove all gradients for this variable
                if freeze_indices == True:
                    g = None

                # Process dense gradients
                elif isinstance(g, tf.Tensor):
                    print("Freezing {} indicies of variable '{}' [D]"
                          .format(len(freeze_indices), v.name))

                    update_shape = [len(freeze_indices)] + list(g.get_shape()[1:])
                    gradient_mask = tf.zeros(update_shape, dtype=g.dtype)
                    g = tf.scatter_mul(g, freeze_indices, gradient_mask)

                # Process sparse gradients
                elif isinstance(g, tf.IndexedSlices):
                    print("Freezing {} indicies of variable '{}' [S]"
                          .format(len(freeze_indices), v.name))

                    # Remove frozen indices from gradient
                    g = tf.sparse_mask(g, freeze_indices)

            if train_vars and (v in train_vars):
                trainable_indices = train_vars[v]

                # Process dense gradients
                if isinstance(g, tf.Tensor):
                    print("Training only on {} indicies of variable '{}' [D]"
                          .format(len(freeze_indices), v.name))

                    gradient_mask = tf.scatter_nd(
                        tf.reshape(trainable_indices, [-1, 1]),
                        tf.ones(tf.get_shape(trainable_indices)),
                        [g.get_shape()[0], 1])
                    g = tf.multiply(g, gradient_mask)

                # Process sparse gradients
                elif isinstance(g, tf.IndexedSlices):
                    print("Training only on {} indicies of variable '{}' [S]"
                          .format(len(freeze_indices), v.name))
                    raise RuntimeError

            processes_grads_and_vars.append((g, v))

        train = optimizer.apply_gradients(processes_grads_and_vars,
                                          global_step=self.global_step,
                                          name="train")

        tf.summary.scalar("Learning rate", lr)
        return train

    def build_graph(self, unigrams):
        """
        Build the graph for the full model.

        Args:
            unigrams: sampling distribution for negative samples

        Returns:
            Graph loss and inputs in a dictionary.
        """
        input_size = None  # self.batch_size

        # Input data.
        target_input = tf.placeholder(tf.int32, shape=[input_size])
        context_input = tf.placeholder(tf.int32, shape=[input_size])
        embeddings_shape = [self.vocabulary_size, self.embedding_size]

        batch_size_t = tf.cast(tf.shape(target_input)[0], tf.float32)

        # Variables.
        init_width = 0.5 / self.embedding_size
        embeddings = tf.get_variable("target_embeddings",
                                     shape=embeddings_shape,
                                     initializer=tf.glorot_normal_initializer())
        context_weights = tf.get_variable("context_embeddings",
                                          shape=embeddings_shape,
                                          initializer=tf.glorot_normal_initializer())
        context_biases = tf.get_variable("context_biases",
                                         shape=[self.vocabulary_size],
                                         initializer=tf.zeros_initializer)

        self._model_variables.update(
            [embeddings, context_weights, context_biases]
        )

        # Negative sampling.
        # Note true_classes needs to be tf.int64
        negative_sample_id, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(context_input, tf.int64), [-1, 1]),
            num_true=1,
            num_sampled=self.num_sampled,
            unique=True,
            range_max=self.vocabulary_size,
            distortion=0.75,
            unigrams=unigrams,
        )
        self._negative_sample = negative_sample_id

        with tf.name_scope("postive_pair"):
            # Embeddings for target: [batch_size, emb_dim]
            positive_emb = tf.nn.embedding_lookup(embeddings, target_input, name="pos_emb")
            # Weights for context: [batch_size, emb_dim]
            positive_w = tf.nn.embedding_lookup(context_weights, context_input, name="pos_con_W")
            # Biases for context: [batch_size, 1]
            positive_b = tf.nn.embedding_lookup(context_biases, context_input, name="pos_con_b")

            # True logits: [batch_size, 1]
            pos_logits = tf.reduce_sum(tf.multiply(positive_emb, positive_w), 1) + positive_b

            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(pos_logits), logits=pos_logits)

        with tf.name_scope("negative_pair"):
            # Weights for sampled ids: [num_sampled, emb_dim]
            negative_w = tf.nn.embedding_lookup(context_weights, negative_sample_id,
                                                name="neg_con_w")
            # Biases for sampled ids: [num_sampled, 1]
            negative_b = tf.nn.embedding_lookup(context_biases, negative_sample_id,
                                                name="neg_con_b")

            # Sampled logits: [batch_size, num_sampled]
            negative_b_vec = tf.reshape(negative_b, [-1])
            neg_logits = tf.matmul(positive_emb,
                                   negative_w,
                                   transpose_b=True) + negative_b_vec

            sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_logits), logits=neg_logits)

        with tf.name_scope("loss"):
            # NCE-loss is the sum of the true and noise (sampled words)
            # contributions, averaged over the batch.
            nce_loss = (tf.reduce_sum(true_xent)
                        + tf.reduce_sum(sampled_xent)) / batch_size_t
            tf.summary.scalar("NCE loss", nce_loss)

            normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)

        self._skipgram_graph = {
            "target_input": target_input,
            "context_input": context_input,
            "embeddings": embeddings,
            "normalized_embeddings": normalized_embeddings,
            "context_weights": context_weights,
            "context_biases": context_biases,
            "loss": nce_loss,
        }
        return self._skipgram_graph

    def eval(self, sess, dataset, summary=None):
        sk_graph = self._skipgram_graph

        if self.val_batch_size is None:
            bs = ds.split_sizes[1]
        else:
            bs = self.val_batch_size

        if dataset.epoch_done(bs, split=1):
            dataset.reset_index(split=1)

        batch_data, batch_labels = dataset.generate_batch(bs, split=1)

        feed_dict = {sk_graph["target_input"]: batch_data,
                     sk_graph["context_input"]: batch_labels}

        if summary is None:
            out = sess.run(sk_graph['loss'], feed_dict=feed_dict)
        else:
            out = sess.run([summary, sk_graph['loss']], feed_dict=feed_dict)

        return out

    def save_epoch_time(self, epoch, time):
        with open(os.path.join(FLAGS.base_log_dir, "epoch_time.txt"), "a") as f:
            f.write("{0}, {1}\n".format(epoch, time))

    def train(self, sess, dataset,
              analogy_dataset=None,
              freeze_indices=None,
              freeze_context_indices=None,
              restore_from_file=None,
              n_epochs=10):
        """
        Train the model on specified data.

        Args:
            sess: Tensorflow session
            dataset: Dataset class (Skipgram)
            analogy_dataset: Evaluation dataset (Analogy)
            n_epochs: Number of epochs to train

        Returns:
            Only emptiness
        """
        sk_graph = self.build_graph(ds.unigrams)

        freeze_vars = None
        if freeze_indices is not None:
            print("Setting freeze vars to embeddings...")
            freeze_vars = {
                sk_graph['embeddings']: list(set(freeze_indices))
            }

        if freeze_context_indices is not None:
            print("Setting freeze vars to context weights and biases...")
            freeze_vars = {
                sk_graph['context_weights']: list(set(freeze_context_indices)),
                sk_graph['context_biases']: list(set(freeze_context_indices))
            }

        with tf.name_scope("train"):
            optimize_fn = self.optimize_graph(sk_graph['loss'], freeze_vars)
        sk_graph['train'] = optimize_fn

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.save_path,
                                               sess.graph)

        # Saver for variables
        saver = tf.train.Saver(list(self._model_variables), max_to_keep=FLAGS.num_checkpoints)

        # Initialize other variables
        init_vars = [v for v in tf.global_variables()
                     if v not in self._model_variables]

        # Restore variables from checkpoint
        if restore_from_file:
            print("Restoring variables from {}...".format(restore_from_file))
            saver.restore(sess, restore_from_file)
            sess.run(tf.variables_initializer(init_vars))

        else:
            # Properly initialize all variables.
            print("No checkpoint file is given. Initializing variables...")
            sess.run(tf.global_variables_initializer())

        ev_ii = -1
        ana_ii = -1
        batch_ii = 0
        for epoch in range(n_epochs):
            # Start new epoch
            ds.reset_index(split=0)

            batch_index = 0
            batch_time = time.time()
            epoch_start = time.time()
            while not dataset.epoch_done(self.batch_size):
                with timeit("generate_batch"):
                    batch_data, batch_labels = dataset.generate_batch(self.batch_size)

                feed_dict = {sk_graph["target_input"]: batch_data,
                             sk_graph["context_input"]: batch_labels}

                with timeit("run"):
                    _, loss_ii = sess.run([sk_graph["train"], sk_graph["loss"]],
                                          feed_dict=feed_dict)

                if batch_ii % 10000 == 0:
                    # Save checkpoint
                    saver.save(sess,
                               os.path.join(self.save_path, "checkpoint"),
                               global_step=self.global_step)

                batch_ii += 1

            epoch_time = time.time() - epoch_start
            print("\nEpoch done in {:4f}s".format(epoch_time))
            self.save_epoch_time(epoch, epoch_time)

            node_embeddings = session.run(sk_graph["normalized_embeddings"])
            self.save_embeddings(epoch, node_embeddings)

            # Save checkpoint
            # Increase the number of checkpoints to hold
            saver.save(sess, os.path.join(self.save_path, "model-epoch"), global_step=epoch)


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('train_split', 1.0, 'train split.')
flags.DEFINE_float('learning_rate', 0.2, 'initial learning rate.')
flags.DEFINE_string('train_prefix', '',
                    'name of the object file that stores the training data. must be specified.')

flags.DEFINE_integer('embedding_size', 20, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('vocab_size', 10400, 'Size of vocabulary.')
flags.DEFINE_integer('n_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('neg_sample_size', 2, 'number of negative samples')
flags.DEFINE_integer('batch_size', 20, 'minibatch size.')
flags.DEFINE_boolean('freeze_embeddings', False,
                     'If true, the embeddings will be frozen otherwise the contexts will be frozen.')

flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('input_dir', '.', 'Input data directory.')
flags.DEFINE_string('train_file', None, 'Input train file name.')
flags.DEFINE_string('label_file', None, 'Input label file name.')
flags.DEFINE_string('degrees_file', None, 'Input node degrees file name.')
flags.DEFINE_string('degrees_dir', None, 'Input node degrees directory.')
flags.DEFINE_string('checkpoint_file', None, 'Input tf checkpoint file name.')
flags.DEFINE_string('checkpoint_dir', None, 'Input tf checkpoint file directory.')
flags.DEFINE_integer('num_checkpoints', None, "Number of checkpoints to keep.")
flags.DEFINE_string('affected_vertices_file', None, 'Input affected vertices file name.')
flags.DEFINE_string('delimiter', '\t', 'Delimiter.')
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('force_offset', 0, "Offset to adjust node IDs.")
flags.DEFINE_integer('seed', 58125312, "Seed for random generator.")

if __name__ == "__main__":
    delimiter = FLAGS.delimiter
    if FLAGS.delimiter == r'\t':
        print("TAB separated")
        delimiter = "\t"

    ds = PregeneratedDataset(os.path.join(FLAGS.input_dir, FLAGS.train_file),
                             n_nodes=FLAGS.vocab_size,
                             delimiter=delimiter,
                             force_offset=FLAGS.force_offset,
                             splits=[FLAGS.train_split, 1 - FLAGS.train_split])

    # We need to set the corresponding graph, in particular use the degree
    # to control the negative sampling, as in word2vec paper
    ds.set_node_degrees(os.path.join(FLAGS.degrees_dir, FLAGS.degrees_file))

    # Set the affected vertices to freeze
    if FLAGS.affected_vertices_file is not None:
        ds.set_affected_nodes(
            os.path.join(FLAGS.degrees_dir, FLAGS.affected_vertices_file))
    else:
        ds.affected_nodes = ds.existing_vocab

    word2vec = W2V_Sampled(
        embedding_size=FLAGS.embedding_size,
        vocabulary_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        val_batch_size=None,
        neg_samples=FLAGS.neg_sample_size,
        save_path=FLAGS.base_log_dir,
        learning_rate=FLAGS.learning_rate
    )

    freeze_context_indices = None
    freeze_indices = None

    if FLAGS.freeze_embeddings:
        freeze_indices = ds.build_freeze_indices()
    else:
        freeze_context_indices = ds.build_freeze_indices()

    with tf.Session() as session, tf.device('/cpu:0'):
        tf.set_random_seed(FLAGS.seed)
        checkpoint_file = None
        if FLAGS.checkpoint_file is not None:
            checkpoint_file = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint_file)
        word2vec.train(session, ds,
                       freeze_indices=freeze_indices,
                       freeze_context_indices=freeze_context_indices,
                       restore_from_file=checkpoint_file,
                       n_epochs=FLAGS.n_epochs)
