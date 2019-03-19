import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from tensorflow.contrib.tensorboard.plugins import projector

class LSTM:

    @staticmethod
    def init_weights(Mi, Mo):
        return np.random.rand(Mi, Mo) / np.sqrt(Mi + Mo)

    def __init__(self, D, M, V, K, batch_size=2, learning_rate=0.05):
        """
        D: dimensionality of word embeddings
        M: size of hidden layer
        V: size of vocabulary
        K: num of output classes
        """
        self.D = D
        self.M = M
        self.V = V
        self.K = K
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        tf.reset_default_graph()

        self.tf_session = tf.Session()

        with tf.name_scope("We"):
            self.We = tf.Variable(tf.random_uniform([self.V, self.D], -0.0001, 0.0001), name="We")

            tf.summary.histogram("hist_We", self.We)

        with tf.name_scope("f_gate"):
            self.Wxf = tf.Variable(LSTM.init_weights(self.D, self.M), dtype=tf.float32, name="Wxf")
            self.Whf = tf.Variable(LSTM.init_weights(self.M, self.M), dtype=tf.float32, name="Whf")

            tf.summary.histogram("hist_Wxf", self.Wxf)
            tf.summary.histogram("hist_Whf", self.Whf)

        with tf.name_scope("i_gate"):
            self.Wxi = tf.Variable(LSTM.init_weights(self.D, self.M), dtype=tf.float32, name="Wxi")
            self.Whi = tf.Variable(LSTM.init_weights(self.M, self.M), dtype=tf.float32, name="Whi")

            tf.summary.histogram("hist_Wxi", self.Wxi)
            tf.summary.histogram("hist_Whi", self.Whi)

        with tf.name_scope("o_gate"):
            self.Wxo = tf.Variable(LSTM.init_weights(self.D, self.M), dtype=tf.float32, name="Wxo")
            self.Who = tf.Variable(LSTM.init_weights(self.M, self.M), dtype=tf.float32, name="Who")

            tf.summary.histogram("hist_Wxo", self.Wxo)
            tf.summary.histogram("hist_Who", self.Who)

        with tf.name_scope("c_hat"):
            self.Wxc = tf.Variable(LSTM.init_weights(self.D, self.M), dtype=tf.float32, name="Wxc")
            self.Whc = tf.Variable(LSTM.init_weights(self.M, self.M), dtype=tf.float32, name="Whc")

            tf.summary.histogram("hist_Wxc", self.Wxc)
            tf.summary.histogram("hist_Whc", self.Whc)

        with tf.name_scope("biases"):
            self.bi = tf.Variable(tf.zeros(shape=[self.M]), dtype=tf.float32, name="bi")
            self.bo = tf.Variable(tf.zeros(shape=[self.M]), dtype=tf.float32, name="bo")
            self.bf = tf.Variable(tf.zeros(shape=[self.M]), dtype=tf.float32, name="bf")
            self.bc = tf.Variable(tf.zeros(shape=[self.M]), dtype=tf.float32, name="bc")

            tf.summary.histogram("hist_bi", self.bi)
            tf.summary.histogram("hist_bo", self.bo)
            tf.summary.histogram("hist_bf", self.bf)
            tf.summary.histogram("hist_bc", self.bc)

        with tf.name_scope("c_0"):
            self.c0 = tf.zeros(shape=[self.M], dtype=tf.float32, name="c0")

        with tf.name_scope("h_0"):
            self.h0 = tf.zeros(shape=[self.M], dtype=tf.float32, name="h0")

        self._initial_hidden_cell_states = tf.stack([self.h0, self.c0])

        with tf.name_scope("output_layer"):
            self.W_op = tf.Variable(LSTM.init_weights(self.M, self.K), dtype=tf.float32, name="W_op")
            self.b_op = tf.Variable(tf.zeros(shape=[self.K]), dtype=tf.float32, name="b_op")

            tf.summary.histogram("hist_W_op", self.W_op)
            tf.summary.histogram("hist_b_op", self.b_op)

        with tf.name_scope("inputs"):
            self.input_seq = tf.placeholder(tf.int32, shape=[None, None], name="input_seq")
            self.targets = tf.placeholder(tf.int32, shape=[None], name="targets")

        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        self.current_batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")
        self.batch_max_len = tf.placeholder(tf.int32, shape=(), name="batch_max_len")

        self.build_graph()

        self.save_dir = "train_log/sentiment_lstm_mini_batch_gd"
        self.model_path = os.path.join(self.save_dir, "model_sentiment_lstm.ckpt")
        self.emb_path = os.path.join(self.save_dir, "word_embedding_sentiment_lstm.npy")
        self.saver = tf.train.Saver(max_to_keep=2)

        self.add_summary_file_writer()
        self.add_summary_embedding()
        self.train_writer.add_graph(graph=self.tf_session.graph, global_step=1)

    def build_graph(self):
        input_embeddings = self.get_embeddings(self.input_seq)

        tensor_array_py_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                           infer_shape=False, name="tensor_array_ho")

        loop_batch_cond = lambda tensor_array_py_x, input_embeddings, idx_sent: tf.less(idx_sent,
                                                                                        self.current_batch_size)
        batch_py_x, _, _ = tf.while_loop(
            loop_batch_cond, self.loop_batch, (tensor_array_py_x, input_embeddings, 0), parallel_iterations=5,
            name="loop_"
        )

        self.py_x = batch_py_x.concat()

        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.targets)
            self.loss = tf.divide(tf.reduce_sum(loss), tf.cast(self.current_batch_size, tf.float32))

        with tf.name_scope("train_op"):
            trainables = tf.trainable_variables()

            grads = tf.gradients(self.loss, trainables)

            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1)
            grad_var_pairs = zip(grads, trainables)

            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            self.train_op = opt.apply_gradients(grad_var_pairs)

    def add_summary_file_writer(self):
        print "Creating FileWriter"
        self.train_writer = tf.summary.FileWriter(self.save_dir, graph=self.tf_session.graph)

    def add_summary_embedding(self):
        """
        refer https://www.tensorflow.org/how_tos/embedding_viz/
        """
        print "Creating Embedding Projections"
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.We.name
        embedding.metadata_path = "sentiment_lstm_vocab.tsv"
        projector.visualize_embeddings(self.train_writer, config)

    def get_embeddings(self, idx_input_seq):
        return tf.nn.embedding_lookup(self.We, idx_input_seq)

    def _recurrence(self, previous_hidden_memory_tuple, x_t):
        h_t_minus_1, c_t_minus_1 = tf.unstack(previous_hidden_memory_tuple)

        x_t = tf.reshape(x_t, [1, self.D])
        h_t_minus_1 = tf.reshape(h_t_minus_1, [1, self.M])
        c_t_minus_1 = tf.reshape(c_t_minus_1, [1, self.M])

        f_t = tf.nn.sigmoid(
            tf.matmul(x_t, self.Wxf) + tf.matmul(h_t_minus_1, self.Whf) + self.bf
        )

        i_t = tf.nn.sigmoid(
            tf.matmul(x_t, self.Wxi) + tf.matmul(h_t_minus_1, self.Whi) + self.bi
        )

        c_hat_t = tf.nn.tanh(
            tf.matmul(x_t, self.Wxc) + tf.matmul(h_t_minus_1, self.Whc) + self.bc
        )

        c_t = (f_t * c_t_minus_1) + (i_t * c_hat_t)

        o_t = tf.nn.sigmoid(
            tf.matmul(x_t, self.Wxo) + tf.matmul(h_t_minus_1, self.Who) + self.bo
        )

        h_t = o_t * tf.nn.tanh(c_t)

        h_t = tf.reshape(h_t, [self.M])
        c_t = tf.reshape(c_t, [self.M])

        return tf.stack([h_t, c_t])

    def loop_batch(self, tensor_array_py_x, input_embeddings, idx_sent):
        hidden_cell_states = tf.scan(
            fn=self._recurrence, elems=tf.gather(input_embeddings, idx_sent),
            initializer=self._initial_hidden_cell_states, name="hidden_states"
        )

        h_t, c_t = tf.unstack(hidden_cell_states, axis=1)

        h_t_last = tf.reshape(h_t[-1, :], [1, self.M])

        py_x = tf.matmul(h_t_last, self.W_op) + self.b_op

        tensor_array_py_x = tensor_array_py_x.write(idx_sent, py_x)
        idx_sent = tf.add(idx_sent, 1)

        return tensor_array_py_x, input_embeddings, idx_sent

    def fit(self, X, Y, lr=1e-2, epochs=500):
        self.learning_rate = lr

        print ("Initializing global variables")
        self.tf_session.run(tf.global_variables_initializer())
        print ("# of trainable var outside: " + str(len(tf.trainable_variables())))

        net_epoch_step_idx = 0

        num_samples = len(X)
        costs = list()
        for idx_epoch in xrange(epochs):
            cost_epoch = 0
            accuracy_epoch = list()

            X_train, Y_train = shuffle(X, Y, n_samples=num_samples)

            net_epoch_step_idx = num_samples * idx_epoch
            current_idx_sent = 0

            while current_idx_sent < len(X_train):
                print ("----------------------------------------------------")
                print ("epoch: {}, sentence: {}".format(idx_epoch, current_idx_sent))

                seq_len = list()

                targets = Y_train[current_idx_sent: current_idx_sent + self.batch_size]

                x = X_train[current_idx_sent: current_idx_sent + self.batch_size]
                max_len = max([len(sentence) for sentence in x])

                input_seq = list()
                for index, sentence in enumerate(x):
                    seq_len.append(len(sentence))
                    padded_sentence = list(
                        np.pad(sentence, (0, max_len - len(sentence)), 'constant', constant_values=0))
                    input_seq.append(padded_sentence)

                current_batch_size = len(seq_len)

                net_epoch_step_idx += current_batch_size

                feed_dict = {
                    self.input_seq: input_seq,
                    self.targets: targets,
                    self.current_batch_size: current_batch_size,
                    self.seq_len: seq_len,
                    self.batch_max_len: max_len
                }

                self.tf_session.run(self.train_op, feed_dict=feed_dict)

                py_x, loss, We = self.tf_session.run([self.py_x, self.loss, self.We], feed_dict=feed_dict)

                pred = np.argmax(py_x, axis=1)
                print "Y: ", targets
                print "Prediction: ", pred

                accuracy = 0
                for y, y_ in zip(targets, pred):
                    if y == y_:
                        accuracy += 1
                accuracy = float(accuracy) / len(pred)

                print "Accuracy/Batch # {}".format(current_idx_sent), accuracy
                print "Loss/Batch # {}".format(current_idx_sent), loss
                cost_epoch += loss

                accuracy_epoch.append(accuracy)

                loss_step = tf.Summary(
                    value=[
                        tf.Summary.Value(tag="loss_per_mini_batch", simple_value=loss),
                    ]
                )

                accuracy_step = tf.Summary(
                    value=[
                        tf.Summary.Value(tag="accuracy_per_mini_batch", simple_value=accuracy),
                    ]
                )

                self.train_writer.add_summary(loss_step, net_epoch_step_idx)
                self.train_writer.add_summary(accuracy_step, net_epoch_step_idx)

                if net_epoch_step_idx % 1000 == 0:
                    self.save_model(step=net_epoch_step_idx)
                    summary = self.tf_session.run(tf.summary.merge_all())
                    self.train_writer.add_summary(summary, idx_epoch)
                    self.train_writer.flush()

                current_idx_sent += self.batch_size

            costs.append(cost_epoch)

            print ("---------")
            print ("Cost at epoch {} is {}".format(idx_epoch, cost_epoch))

            print ("Accuracy at epoch {} is {}".format(idx_epoch, np.mean(accuracy_epoch)))
            print ("---------")

            self.save_model(step=net_epoch_step_idx)

            summary_cost_epoch = tf.Summary(
                value=[
                    tf.Summary.Value(tag="loss/epoch", simple_value=cost_epoch),
                ]
            )

            summary_accuracy_epoch = tf.Summary(
                value=[
                    tf.Summary.Value(tag="accuracy/epoch", simple_value=np.mean(accuracy_epoch)),
                ]
            )

            self.train_writer.add_summary(summary_cost_epoch, idx_epoch)
            self.train_writer.add_summary(summary_accuracy_epoch, idx_epoch)
            summary = self.tf_session.run(tf.summary.merge_all())
            self.train_writer.add_summary(summary, idx_epoch)
            self.train_writer.flush()

    def save_model(self, step=1):
        self.save_embedding_matrix()
        print "saving model for step", step, "to", self.model_path
        self.saver.save(self.tf_session, self.model_path, step)

    def save_embedding_matrix(self):
        np.save(self.emb_path, self.We.eval(self.tf_session))

    def close_session(self):
        self.tf_session.close()