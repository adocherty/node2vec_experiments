#
# Node2Vec example
#  Code modified from tensorflow models tutorial
#  [link]
#

import datetime
import os
import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
import pickle
import tensorflow as tf
import networkx as nx

from sklearn import model_selection, linear_model, metrics, svm

times = {}
@contextmanager
def timeit(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    times[name] = elapsedTime + times.get(name, 0)

class PregeneratedDataset:
    def __init__(self, data_filename, n_nodes, delimiter=" ", force_offset=0, splits=[0.5,0.5]):

        self.splits = splits
        self.n_splits = len(splits)
        self.data_index = [0]*self.n_splits
        self.split_sizes = []
        self.split_data = []
        self.vocab_size = int(n_nodes)
        self.labels = None

        self.build_dataset(data_filename, delimiter, force_offset)

    def set_node_degrees(self, degree):
        self.unigrams = [degree[ii] for ii in range(self.vocab_size)]

    def epoch_done(self, batch_size=0, split=0):
        return self.data_index[split] + batch_size > self.split_sizes[split]

    def reset_index(self, split=0):
        self.data_index[split] = 0

    def load_labels(self, label_filename, delimiter=" ", force_offset=0):
        raw_labels = pd.read_csv(label_filename, delimiter=delimiter,
                                dtype='int32', header=None).values

        self.labels = np.zeros(self.vocab_size, dtype=np.int32)
        for (index, label) in raw_labels:
            self.labels[index + force_offset] = label

    def build_dataset(self, data_filename, delimiter=" ", force_offset=0):
        """Process raw inputs into a dataset."""

        # Load all data
        print("Loading target-context pairs from {}".format(data_filename))
        self.data = pd.read_csv(data_filename, delimiter=delimiter,
                                dtype='int32', header=None).values

        # Force an adjustment to the node indices
        self.data += force_offset

        n_total = len(self.data)
        self.split_sizes = [int(n_total * split) for split in self.splits]
        self.split_offset = [0] + self.split_sizes[:-1]
        self.data_index = [0] * self.n_splits

    def generate_batch(self, batch_size, split=0):
        """
        Generate data as (target_word, context_word) pairs.
        """
        data_size = self.split_sizes[split]
        data_offset = self.data_index[split] + self.split_offset[split]

        # Variable batch size - ensure model can handle this
        batch_size = min(batch_size, data_size - self.data_index[split])

        batch = np.empty((batch_size), dtype=np.int32)
        labels = np.empty((batch_size), dtype=np.int32)

        batch[:] = self.data[data_offset : data_offset + batch_size, 0]
        labels[:] = self.data[data_offset : data_offset + batch_size, 1]

        self.data_index[split] += batch_size

        return batch, labels


class AnalogyDataset:
    def __init__(self, filename, word_to_index):
        self.analogy_file = filename
        self.word_to_index = word_to_index
        self.build_dataset()

    def build_dataset(self):
        """Reads through the analogy question file.

          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
        """
        questions = []
        questions_skipped = 0
        with open(self.analogy_file, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(b" ")

                ids = [self.word_to_index.get(w.strip().decode()) for w in words]

                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(ids)

        print("Eval analogy file: ", self.analogy_file)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)

        self.analogy_questions = np.array(questions, dtype=np.int32)


class W2V_Sampled:
    def __init__(self, embedding_size, vocabulary_size,
                 batch_size=100,
                 val_batch_size=None,
                 save_path=None,
                 learning_rate=0.2,
                 neg_samples=64,
                 lr_decay=0.1):

        self.vocabulary_size = vocabulary_size   #
        self.batch_size = batch_size             #
        self.val_batch_size = val_batch_size

        self.embedding_size = embedding_size     # Embeddings
        self.analogy_k = 4
        self.num_sampled = neg_samples           # Number of negative examples to sample.
        self.learning_rate = learning_rate       # Initial learning rate
        self.lr_decay = lr_decay * 1e-6          # LR exponential decay
        self.save_path = save_path               # Where to store TF output

        self._model_variables = set()

    def save_embeddings(self):
        pass

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
        input_size = None # self.batch_size

        # Input data.
        target_input = tf.placeholder(tf.int32, shape=[input_size])
        context_input = tf.placeholder(tf.int32, shape=[input_size])
        embeddings_shape = [self.vocabulary_size, self.embedding_size]

        batch_size_t = tf.cast(tf.shape(target_input)[0], tf.float32)

        # Variables.
        embeddings = tf.get_variable("target_embeddings",
                                     shape=embeddings_shape,
                                     initializer=tf.glorot_normal_initializer())
        context_weights = embeddings
        # context_weights = tf.get_variable("context_embeddings",
        #                              shape=embeddings_shape,
        #                              initializer=tf.glorot_normal_initializer())
        context_biases = tf.get_variable("context_biases",
                                     shape=[self.vocabulary_size],
                                     initializer=tf.zeros_initializer)

        self._model_variables.update(
            [embeddings, context_weights, context_biases]
        )

        # Freeze some Weights - Note this slows things down 100x!
        # freeze_indices = list(freeze_indices)
        # train_indices = [x for x in range(self.vocabulary_size)
        #                  if x not in freeze_indices]
        #
        # freeze_emb = tf.nn.embedding_lookup(embeddings, freeze_indices)
        # train_emb = tf.nn.embedding_lookup(embeddings, train_indices)
        # freeze_emb_s = tf.scatter_nd(tf.reshape(freeze_indices, [-1,1]),
        #                              freeze_emb, tf.shape(embeddings))
        # train_emb_s = tf.scatter_nd(tf.reshape(train_indices, [-1,1]),
        #                             train_emb, tf.shape(embeddings))
        # embeddings = tf.stop_gradient(freeze_emb_s) + train_emb_s

        # Negative sampling.
        # Note true_classes needs to be tf.int64
        negative_sample_id, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(tf.cast(context_input, tf.int64), [-1,1]),
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
            negative_w = tf.nn.embedding_lookup(context_weights, negative_sample_id, name="neg_con_w")
            # Biases for sampled ids: [num_sampled, 1]
            negative_b = tf.nn.embedding_lookup(context_biases, negative_sample_id, name="neg_con_b")

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
                        + tf.reduce_sum(sampled_xent))/batch_size_t
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

    def build_nearest_graph(self):
        sk_graph = self._skipgram_graph

        with tf.name_scope("find_nearest"):
            nemb = tf.nn.l2_normalize(sk_graph["embeddings"], 1)

            nearby_word = tf.placeholder(dtype=tf.int32, name='nearby_in')
            nearby_emb = tf.gather(nemb, nearby_word)
            nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
            nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, 5)

        self._nearby_graph = {
            'input_word': nearby_word,
            'nearby_index': nearby_idx,
            'nearby_val': nearby_val,
        }
        return self._nearby_graph

    def build_analogy_graph(self):
        """Graph for analogy prediction:

        Each analogy task is to predict the 4th word (d) given three
        words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        predict d=paris.
        """
        sk_graph = self._skipgram_graph
        with tf.name_scope("analogy"):
            # Predict d from (a,b,c)
            # using the embedding algebra d = c + (b - a)
            analogy_a = tf.placeholder(dtype=tf.int32, name="ana_a")  # [N]
            analogy_b = tf.placeholder(dtype=tf.int32, name="ana_b")  # [N]
            analogy_c = tf.placeholder(dtype=tf.int32, name="ana_c")  # [N]

            # Normalized word embeddings of shape [vocab_size, emb_dim]
            nemb = sk_graph["normalized_embeddings"]

            # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
            # They all have the shape [N, emb_dim]
            a_emb = tf.gather(nemb, analogy_a)
            b_emb = tf.gather(nemb, analogy_b)
            c_emb = tf.gather(nemb, analogy_c)

            # We expect that d's embedding vectors on the unit hyper-sphere is
            # near: c_emb + (b_emb - a_emb), shape: [N, emb_dim]
            target = c_emb + (b_emb - a_emb)

            # Compute cosine distance between each pair of target and vocab.
            # shape [N, vocab_size]
            dist = tf.matmul(target, nemb, transpose_b=True)

            # For each question (row in dist), find the top k words.
            _, pred_idx = tf.nn.top_k(dist, self.analogy_k)

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_graph = {
            "a": analogy_a,
            "b": analogy_b,
            "c": analogy_c,
            "predict": pred_idx,
        }

    def eval_analogy(self, sess, ad):
        # The TF variables for the analogy graph
        tfvar = self._analogy_graph
        total = ad.analogy_questions.shape[0]

        start = 0
        correct = 0
        while start < total:
            limit = start + 2500
            analogy = ad.analogy_questions[start:limit, :]

            feed_dict = {
                tfvar["a"]: analogy[:, 0],
                tfvar["b"]: analogy[:, 1],
                tfvar["c"]: analogy[:, 2],
            }
            pred_idx = sess.run(tfvar["predict"], feed_dict)

            start = limit
            for ii in range(analogy.shape[0]):
                for jj in range(self.analogy_k):
                    if pred_idx[ii, jj] == analogy[ii, 3]:
                        correct += 1
                        break
                    elif pred_idx[ii, jj] in analogy[ii, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break

        # print("Eval %4d/%d accuracy = %4.1f%%"%(correct, total,
        #                                         correct*100.0/total))
        return correct/total

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

    def eval_nearby(self, sess, dataset, ids, num=20):
        """Prints out nearby IDs given a list of IDs."""
        nb_graph = self._nearby_graph

        nidx, nval = sess.run(
            [nb_graph['nearby_index'],nb_graph['nearby_val']],
            {nb_graph['input_word']: ids}
            )

        for ii,word_id in enumerate(ids):
            print("\n=====================================")
            print(word_id)
            for neighbor, distance in zip(nidx[ii], nval[ii]):
                print("%-20s %6.4f" % (neighbor, distance))

    def eval_classification(self, session, labels, train_size):
        sk_graph = self._skipgram_graph
        node_embeddings = session.run(sk_graph["normalized_embeddings"])

        # Classifier choice
        classifier = linear_model.LogisticRegression(C=10)
        #classifier = svm.SVC(C=1)

        scoring = ['accuracy', 'f1_macro', 'f1_micro']

        shuffle = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.8)

        cv_scores = model_selection.cross_validate(
            classifier, node_embeddings, labels,
            scoring=scoring, cv=shuffle, return_train_score=True
        )
        train_acc = cv_scores['train_accuracy'].mean()
        train_f1 = cv_scores['train_f1_macro'].mean()
        test_acc = cv_scores['test_accuracy'].mean()
        test_f1 = cv_scores['test_f1_macro'].mean()

        print("Train acc: {:0.3f}, f1: {:0.3f}"
              .format(train_acc, train_f1))
        print("Test acc: {:0.3f}, f1: {:0.3f}"
              .format(test_acc, test_f1))

        return {'train_acc': train_acc, 'test_acc': test_acc, 'train_f1': train_f1, 'test_f1': test_f1}

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
            freeze_vars = {
                sk_graph['embeddings']: list(freeze_indices)
            }

        if freeze_context_indices is not None:
            freeze_vars = {
                sk_graph['context_weights']: list(freeze_context_indices),
                sk_graph['context_biases']: list(freeze_context_indices)
            }

        with tf.name_scope("train"):
            optimize_fn = self.optimize_graph(sk_graph['loss'], freeze_vars)
        sk_graph['train'] = optimize_fn

        # Build graph for analogy evaluation
        if analogy_dataset:
            self.build_analogy_graph()

        # Graph to find closest words by embedding
        self.build_nearest_graph()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.save_path,
                                               sess.graph)

        # Saver for variables
        saver = tf.train.Saver(list(self._model_variables))

        # Initialize other variables
        init_vars = [v for v in tf.global_variables()
                     if v not in self._model_variables]

        # Restore variables from checkpoint
        if restore_from_file:
            print("Restoring variables from {}".format(restore_from_file))
            saver.restore(sess, restore_from_file)
            sess.run(tf.variables_initializer(init_vars))

        else:
            # Properly initialize all variables.
            sess.run(tf.global_variables_initializer())

        ev_ii = -1
        ana_ii = -1
        batch_ii = 0
        for epoch in range(n_epochs):
            # Start new epoch
            ds.reset_index(split=0)

            if dataset.labels is not None:
                print("\nClassification evaluation:")
                self.eval_classification(sess, dataset.labels, 0.2)

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

                if batch_ii % 1000 == 0:
                    # Evaluate and add evaluation info
                    sum_ii, ev_ii = self.eval(sess, dataset, summary=summary_op)
                    summary_writer.add_summary(sum_ii, batch_ii//1000)

                    train_wps = np.floor((dataset.data_index[0] - batch_index)
                                         / (time.time() - batch_time))
                    pc_done = 100.0*dataset.data_index[0] / dataset.split_sizes[0]
                    print("Epoch {} [{:0.1f}%], loss: {:0.1f}, val: {:0.3f}, ana: {:0.2f} word/sec: {:0.0f}  |  "
                        .format(epoch, pc_done, loss_ii, ev_ii, ana_ii, train_wps), end="\r")

                    batch_time = time.time()
                    batch_index = dataset.data_index[0]

                batch_ii += 1

            epoch_time = time.time() - epoch_start
            print("\nEpoch done in {:4f}s".format(epoch_time))

            # Save checkpoint
            saver.save(sess, os.path.join(self.save_path, "model_epoch_"), global_step=epoch)


if __name__ == "__main__":
    karate = nx.karate_club_graph()

    ds = PregeneratedDataset("gPairs-w3-s6.txt",
                             n_nodes=karate.number_of_nodes(),
                             delimiter="\t",
                             force_offset=-1,
                             splits=[0.8,0.2])

    # We need to set the corresponding graph, in particular use the degree
    # to control the negative sampling, as in node2vec paper
    ds.set_node_degrees(karate.degree())

    # Set labels
    ds.load_labels("karate-labels.txt", delimiter="\t", force_offset=-1)

    word2vec = W2V_Sampled(
        embedding_size=20,
        vocabulary_size=ds.vocab_size,
        batch_size=20,
        val_batch_size=None,
        neg_samples=2,
        save_path="n2v_{}".format(datetime.date.today()),
        learning_rate=0.2
        )

    # freeze_context_indices = [199, 200, 399, 400]
    # freeze_indices = None
    # checkpoint_file = "n2v_2018-04-18/checkpoint-170"

    freeze_context_indices = None
    freeze_indices = None
    checkpoint_file = None

    with tf.Session() as session, tf.device('/cpu:0'):
        tf.set_random_seed(58125312)

        word2vec.train(session, ds,
                       freeze_indices=freeze_indices,
                       freeze_context_indices=freeze_context_indices,
                       restore_from_file=checkpoint_file,
                       n_epochs=3)


