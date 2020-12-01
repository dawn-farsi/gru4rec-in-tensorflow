import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.activations import elu
import time
from services import TensorflowVariables
from services.CustomOptimizer import RMSPropWithMomentum


class GruBase(object):
    def __init__(self, is_straining, session_key, item_key, time_key, batch_size, embedding=0,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 hidden_act='tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100,
                 n_epochs=10,
                 dropout_p_hidden=0, learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0, dropout_p_embed=0):
        """
        :param session_key: string, header of the session ID column in the input file (default: 'SessionId')
        :param item_key: string, header of the item ID column in the input file (default: 'ItemId')
        :param time_key: string
        :param layers: list of int values, list of the number of GRU units in the layers (default : [100])
        :param batch_size : int, size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 32)
        :param pre_embedding_y: int, size of the embedding used for output
        :param pre_embedding_x: int, size of the embedding used for input
        :param embedding : int, size of the embedding used, 0 means not to use embedding (default: 0)
        :param initializer: tensorflow random normal initializer
        :param adapt: string, sets the appropriate learning rate adaptation strategy, (default: "adagrad")
        :param hidden_act: string, 'linear', 'relu', 'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function on the hidden states, <X> and <Y> are the parameters of the activation function (default : 'tanh')
        :param final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function of the final layer, <X> and <Y> are the parameters of the activation function (default : 'elu-1')
        :param dropout_p_embed : float, probability of dropout of the input units, applicable only if embeddings are used (default: 0.0)
        """
        self.is_training = is_straining
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.batch_size = batch_size
        self.embedding = embedding
        self.pre_embedding_x = pre_embedding_x
        self.pre_embedding_y = pre_embedding_y
        self.initializer = initializer

        self.hidden_act = hidden_act
        self.set_hidden_activation(self.hidden_act)
        self.final_act = final_act
        self.set_final_activation(self.final_act)
        self.loss = loss
        self.set_loss_function(loss)
        self.grad_cap = grad_cap
        self.layers = layers
        self.rnn_size = rnn_size
        self.n_epochs = n_epochs
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.dropout_p_embed = dropout_p_embed
        self.n_items = None
        self.itemidmap = None

    def linear(self, inp):
        return inp

    def tanh(self, inp):
        return tf.nn.tanh(inp)

    def softmax(self, inp):
        return tf.nn.softmax(inp)

    def softmaxth(self, inp):
        return tf.nn.softmax(tf.tanh(inp))

    def relu(self, inp):
        return tf.nn.relu(inp)

    def sigmoid(self, inp):
        return tf.nn.sigmoid(inp)

    class Elu:
        def __init__(self, alpha):
            self.alpha = alpha

        def execute(self, inp):
            return elu(inp, self.alpha)

    def bpr(self, yhat, weights=None):
        yhatT = tf.transpose(yhat)
        if weights:
            return tf.reduce_mean(
            -tf.compat.v1.log(tf.nn.sigmoid(tf.expand_dims(tf.linalg.diag_part(yhat), axis=1) - yhatT)) * weights)
        else:
            return tf.reduce_mean(
                -tf.compat.v1.log(tf.nn.sigmoid(tf.expand_dims(tf.linalg.diag_part(yhat), axis=1) - yhatT)) )

    def set_hidden_activation(self, hidden_act):
        if self.hidden_act == 'tanh':
            self.hidden_activation = self.tanh
        elif self.hidden_act == 'relu':
            self.hidden_activation = self.relu
        else:
            raise NotImplementedError

    def set_final_activation(self, final_act):
        if final_act == 'linear':
            self.final_activation = self.linear
        elif final_act == 'relu':
            self.final_activation = self.relu
        elif final_act == 'softmax':
            self.final_activation = self.softmax
        elif final_act == 'tanh':
            self.final_activation = self.tanh
        elif final_act.startswith('elu-'):
            self.final_activation = self.Elu(float(final_act.split('-')[1])).execute
        else:
            raise NotImplementedError

    def set_loss_function(self, loss):
        if loss == 'bpr':
            self.loss_function = self.bpr
        else:
            raise NotImplementedError

    def dropout(self, X, drop_p):
        # tf.random.set_seed(0)
        if drop_p > 0:
            X = tf.nn.dropout(X, drop_p, seed=1)
        return X


class Gru4rec(GruBase):
    def __init__(self, is_straining, session_key, item_key, time_key, batch_size, embedding=0,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 hidden_act='tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100,
                 n_epochs=10,
                 dropout_p_hidden=0, learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0, dropout_p_embed=0):
        super().__init__(is_straining, session_key, item_key, time_key, batch_size, embedding,
                         initializer,
                         hidden_act, final_act, loss, grad_cap, layers, rnn_size, n_epochs,
                         dropout_p_hidden, learning_rate, checkpoint_dir,
                         adapt, pre_embedding_y, pre_embedding_x, dropout_p_embed)

    def process_data(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data.astype({'ItemIdx': 'int32'})
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return data, offset_sessions

    def init(self):
        self.Wx, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []
        if self.embedding:
            with tf.name_scope("gru_embedding"):
                self.E = tf.compat.v1.get_variable("E", shape=[self.n_items, self.embedding], initializer=self.initializer,
                                         dtype=tf.float32)
                n_features = self.embedding
        else:
            n_features = self.n_items

        with tf.name_scope("gru_weights"):
            for i in range(len(self.layers)):
                self.Wx.append(tf.compat.v1.get_variable("Wx{}".format(i),
                                               shape=[self.layers[i - 1] if i > 0 else n_features, 3 * self.layers[i]],
                                               initializer=self.initializer, dtype=tf.float32))
                self.Wh.append(tf.compat.v1.get_variable('Wh{}'.format(i), shape=[self.layers[i], self.layers[i]],
                                               initializer=self.initializer, dtype=tf.float32))
                self.Wrz.append(tf.compat.v1.get_variable("Wrz{}".format(i), shape=[self.layers[i], 2 * self.layers[i]],
                                                initializer=self.initializer, dtype=tf.float32))
                self.Bh.append(
                    tf.compat.v1.get_variable("Bh{}".format(i), shape=[3 * self.layers[i]], initializer=tf.zeros_initializer(),
                                    dtype=tf.float32))
                self.H.append(tf.compat.v1.get_variable("H{}".format(i), shape=[self.batch_size, self.layers[i]],
                                              initializer=tf.zeros_initializer(), dtype=tf.float32))
            self.Wy = tf.compat.v1.get_variable("Wt".format(i), shape=[self.n_items, self.layers[-1]],
                                      initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.By = tf.compat.v1.get_variable('By', [self.n_items, 1], initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            # self.global_step = tf.compat.v1.get_variable(0, name='global_step', trainable=False)

    def model(self, X, H, M, R=None, Y=None, drop_p_hidden=0.0, drop_p_embed=0.0, predict=False):
        sparams, full_params, sidxs = [], [], []

        if self.embedding:
            Sx = tf.nn.embedding_lookup(self.E, X)
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
            sparams.append(Sx)
            full_params.append(self.E)
            sidxs.append(X)
        else:
            Sx = tf.nn.embedding_lookup(self.Wx[0], X)
            vec = Sx + self.Bh[0]
            rz = tf.nn.sigmoid(vec[:, self.layers[0]:] + tf.matmul(H[0], self.Wrz[0]))
            h = self.hidden_activation(tf.matmul(H[0] * rz[:, :self.layers[0]], self.Wh[0]) + vec[:, :self.layers[0]])
            z = rz[:, self.layers[0]:]
            h = (1.0 - z) * H[0] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            start = 1
            H_new = [h]
            sparams.append(Sx)
            full_params.append(self.Wx[0])
            sidxs.append(X)
        for i in range(start, len(self.layers)):
            vec = tf.matmul(y, self.Wx[i]) + self.Bh[i]
            rz = tf.nn.sigmoid(vec[:, self.layers[i]:] + tf.matmul(H[i], self.Wrz[i]))
            h = self.hidden_activation(tf.matmul(H[i] * rz[:, :self.layers[i]], self.Wh[i]) + vec[:, :self.layers[i]])
            z = rz[:, self.layers[i]:]
            h = (1.0 - z) * H[i] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            H_new.append(h)
        if Y is not None:
            Sy = tf.nn.embedding_lookup(self.Wy, Y)
            sparams.append(Sy)
            full_params.append(self.Wy)
            sidxs.append(Y)

            SBy = tf.nn.embedding_lookup(self.By, Y)
            sparams.append(SBy)
            full_params.append(self.By)
            sidxs.append(Y)

            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(tf.matmul(y, tf.transpose(Sy.T)) + tf.squeeze(SBy))
            else:
                y = self.final_activation(tf.matmul(y, tf.transpose(Sy)) + tf.squeeze(SBy))
            return H_new, y, sparams, full_params, sidxs
        else:
            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(tf.matmul(y, tf.transpose(self.Wy)) + tf.squeeze(self.By))
            else:
                y = self.final_activation(tf.matmul(y, tf.transpose(self.Wy)) + tf.squeeze(self.By))
            return H_new, y, sparams, full_params, sidxs

    def fit(self, data, offset_sessions, sess):
        self.predict = None
        self.error_during_train = False
        # Setup placeholders
        X = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='input')
        Y = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='output')
        R = tf.compat.v1.placeholder(tf.bool, [self.batch_size], name='reset')
        M = tf.compat.v1.placeholder(tf.float32, [], name='m')
        # Build tensorflow model
        H_new, Y_pred, sparams, full_params, sidxs = self.model(X, self.H, M, R, Y, self.dropout_p_hidden,
                                                                self.dropout_p_embed)
        params = [self.Wx if self.embedding else self.Wx[1:], self.Wh, self.Wrz, self.Bh]
        # Setup cost, optimizer and train_op
        self.cost = (M / self.batch_size) * self.loss_function(Y_pred)
        tvars = tf.compat.v1.trainable_variables()
        optimizer = RMSPropWithMomentum(lr=0.1, epsilon=1e-06, momentum=0.3)
        self.train_op = optimizer.get_updates(self.cost, tvars)
        # Initialize global variable
        sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)
        # Initialize TensorflowVariablses
        self.tf_variables = TensorflowVariables.TensorflowVariables(self.cost, sess)

        for epoch in range(self.n_epochs):
            start_time = time.time()
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(len(self.layers))]
            epoch_cost = []
            session_idx_arr = np.arange(len(offset_sessions) - 1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            new_weights = {}
            while not finished:
                minlen = (end - start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start + i + 1]
                    fetches = [self.cost, self.H, self.train_op]
                    feed_dict = {X: in_idx, Y: out_idx, M: len(iters)}
                    for j in range(len(self.layers)):
                        new_weights[f"H{j}"] = state[j]
                    self.tf_variables.set_weights(new_weights)
                    cost, state, _ = sess.run(fetches, feed_dict)
                    reset = (start + i + 1 == end - 1)
                    if reset.sum() > 0:
                        for i in range(len(self.layers)):
                            state[i][reset] = 0
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                start = start + minlen - 1
                finished_mask = (end - start <= 1)
                n_finished = finished_mask.sum()
                iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
                maxiter += n_finished
                valid_mask = (iters < len(offset_sessions) - 1)
                n_valid = valid_mask.sum()
                if n_valid == 0 or maxiter > len(offset_sessions):
                    finished = True
                    break
                mask = finished_mask & valid_mask
                sessions = session_idx_arr[iters[mask]]
                start[mask] = offset_sessions[sessions]
                end[mask] = offset_sessions[sessions + 1]
                iters = iters[valid_mask]
                start = start[valid_mask]
                end = end[valid_mask]
                for i in range(len(self.layers)):
                    state[i][finished_mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
            end_time = time.time()
            print("cost: {}, time:{}".format(avgc, (end_time - start_time) / 60))
        self.saver.save(sess, '{}/gru4rec-model'.format(self.checkpoint_dir), global_step=epoch)

    def getWeights(self):
        self.weights = self.tf_variables.get_weights()

    def setWeights(self, new_weights):
        self.tf_variables.set_weights(new_weights)

    def symbolic_predict(self, X, Y, M, H, items):
        if items is not None:
            H_new, yhat, _, _, _ = self.model(X, H, M, R=None, Y=Y, predict=True)
        else:
            H_new, yhat, _, _, _ = self.model(X, H, M, R=None, Y=None, predict=True)
        return yhat, H_new

    def variable(self, sess, obj):
        self.tf_variables = TensorflowVariables.TensorflowVariables(obj, sess)

    def getWeights(self):
        self.weights = self.tf_variables.get_weights()

    def setWeights(self, new_weights):
        self.tf_variables.set_weights(new_weights)


class Gru4recWithContext(GruBase):
    def __init__(self, is_straining, session_key, item_key, time_key, batch_size, embedding=0,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 hidden_act='tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100,
                 n_epochs=10,
                 dropout_p_hidden=0, learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0, dropout_p_embed=0):
        super().__init__(is_straining, session_key, item_key, time_key, batch_size, embedding,
                         initializer,
                         hidden_act, final_act, loss, grad_cap, layers, rnn_size, n_epochs,
                         dropout_p_hidden, learning_rate, checkpoint_dir,
                         adapt, pre_embedding_y, pre_embedding_x, dropout_p_embed)

    def process_data(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data.astype({'ItemIdx': 'int32'})
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return data, offset_sessions

    def init(self, n_context_columns):
        self.Wx1, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []
        if self.embedding:
            with tf.name_scope("gru_embedding"):
                self.E = tf.compat.v1.get_variable("E", shape=[self.n_items, self.embedding], initializer=self.initializer,
                                         dtype=tf.float32)
                n_features = self.embedding
        else:
            n_features = self.n_items

        with tf.name_scope("gru_weights"):
            for i in range(len(self.layers)):
                self.Wx1.append(tf.compat.v1.get_variable("Wx1{}".format(i),
                                                shape=[self.layers[i - 1] if i > 0 else n_features, 3 * self.layers[i]],
                                                initializer=self.initializer, dtype=tf.float32))
                self.Wh.append(tf.compat.v1.get_variable('Wh{}'.format(i), shape=[self.layers[i], self.layers[i]],
                                               initializer=self.initializer, dtype=tf.float32))
                self.Wrz.append(tf.compat.v1.get_variable("Wrz{}".format(i), shape=[self.layers[i], 2 * self.layers[i]],
                                                initializer=self.initializer, dtype=tf.float32))
                self.Bh.append(
                    tf.compat.v1.get_variable("Bh{}".format(i), shape=[3 * self.layers[i]], initializer=tf.zeros_initializer(),
                                    dtype=tf.float32))
                self.H.append(tf.compat.v1.get_variable("H{}".format(i), shape=[self.batch_size, self.layers[i]],
                                              initializer=tf.zeros_initializer(), dtype=tf.float32))
            self.Wy = tf.compat.v1.get_variable("Wt".format(i), shape=[self.n_items, self.layers[-1]],
                                      initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.By = tf.compat.v1.get_variable('By', [self.n_items, 1], initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            self.Wyx2 = tf.compat.v1.get_variable("Wyx2", shape=[self.layers[-1], n_context_columns + self.layers[-1]],
                                        initializer=self.initializer, dtype=tf.float32)
            self.Byx2 = tf.compat.v1.get_variable("Byx2", shape=[self.layers[-1], 1], initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)
            # self.global_step = tf.compat.v1.get_variable(0, name='global_step', trainable=False)

    def model(self, X1, X2, H, M, R=None, Y=None, drop_p_hidden=0.0, drop_p_embed=0.0, predict=False):
        sparams, full_params, sidxs = [], [], []

        if self.embedding:
            Sx = tf.nn.embedding_lookup(self.E, X1)
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
            sparams.append(Sx)
            full_params.append(self.E)
            sidxs.append(X1)
        else:
            Sx = tf.nn.embedding_lookup(self.Wx1[0], X1)
            vec = Sx + self.Bh[0]
            rz = tf.nn.sigmoid(vec[:, self.layers[0]:] + tf.matmul(H[0], self.Wrz[0]))
            h = self.hidden_activation(
                tf.matmul(H[0] * rz[:, :self.layers[0]], self.Wh[0]) + vec[:, :self.layers[0]])
            z = rz[:, self.layers[0]:]
            h = (1.0 - z) * H[0] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            start = 1
            H_new = [h]
            sparams.append(Sx)
            full_params.append(self.Wx1[0])
            sidxs.append(X1)
        for i in range(start, len(self.layers)):
            vec = tf.matmul(y, self.Wx1[i]) + self.Bh[i]
            rz = tf.nn.sigmoid(vec[:, self.layers[i]:] + tf.matmul(H[i], self.Wrz[i]))
            h = self.hidden_activation(
                tf.matmul(H[i] * rz[:, :self.layers[i]], self.Wh[i]) + vec[:, :self.layers[i]])
            z = rz[:, self.layers[i]:]
            h = (1.0 - z) * H[i] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            H_new.append(h)
        if Y is not None:
            Sy = tf.nn.embedding_lookup(self.Wy, Y)
            sparams.append(Sy)
            full_params.append(self.Wy)
            sidxs.append(Y)

            SBy = tf.nn.embedding_lookup(self.By, Y)
            sparams.append(SBy)
            full_params.append(self.By)
            sidxs.append(Y)

            if predict and self.final_act == 'softmax_logit':
                yx2 = self.relu(tf.matmul(tf.concat([y, X2], 1), tf.transpose(self.Wyx2)) + tf.squeeze(
                    self.Byx2))
                y = self.softmax(tf.matmul(yx2, tf.transpose(Sy.T)) + tf.squeeze(SBy))
            else:
                yx2 = self.relu(tf.matmul(tf.concat([y, X2], 1), tf.transpose(self.Wyx2)) + tf.squeeze(self.Byx2))
                y = self.final_activation(tf.matmul(yx2, tf.transpose(Sy)) + tf.squeeze(SBy))
            return H_new, y, sparams, full_params, sidxs
        else:
            if predict and self.final_act == 'softmax_logit':
                yx2 = self.relu(tf.matmul(tf.concat([y, X2], 1), tf.transpose(self.Wyx2)) + tf.squeeze(self.Byx2))
                y = self.softmax(tf.matmul(yx2, tf.transpose(self.Wy)) + tf.squeeze(self.By))
            else:
                yx2 = self.relu(tf.matmul(tf.concat([y, X2], 1), tf.transpose(self.Wyx2)) + tf.squeeze(self.Byx2))
                y = self.final_activation(tf.matmul(yx2, tf.transpose(self.Wy)) + tf.squeeze(self.By))
            return H_new, y, sparams, full_params, sidxs

    def fit(self, data, context_columns, n_context_column, offset_sessions, sess):
        self.predict = None
        self.error_during_train = False
        data_context = np.array(data[context_columns].tolist())
        # Setup placeholders
        X1 = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='X1')
        X2 = tf.compat.v1.placeholder(tf.float32, [self.batch_size, n_context_column], name='X2')
        Y = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='Y')
        R = tf.compat.v1.placeholder(tf.bool, [self.batch_size], name='R')
        M = tf.compat.v1.placeholder(tf.float32, [], name='M')
        # Build tensorflow model
        H_new, Y_pred, sparams, full_params, sidxs = self.model(X1, X2, self.H, M, R, Y, self.dropout_p_hidden,
                                                                self.dropout_p_embed)
        # params = [self.Wx1 if self.embedding else self.Wx1[1:], self.Wx2, self.Wh, self.Wrz, self.Bh]
        # Setup cost, optimizer and train_op
        self.cost = (M / self.batch_size) * self.loss_function(Y_pred)
        tvars = tf.compat.v1.trainable_variables()
        optimizer = RMSPropWithMomentum(lr=0.1, epsilon=1e-06, momentum=0.3)
        self.train_op = optimizer.get_updates(self.cost, tvars)
        # Initialize global variable
        sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)
        # Initialize TensorflowVariablses
        self.tf_variables = TensorflowVariables.TensorflowVariables(self.cost, sess)

        for epoch in range(self.n_epochs):
            start_time = time.time()
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(len(self.layers))]
            epoch_cost = []
            session_idx_arr = np.arange(len(offset_sessions) - 1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            new_weights = {}
            while not finished:
                minlen = (end - start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    context = data_context[start + i]
                    out_idx = data.ItemIdx.values[start + i + 1]
                    fetches = [self.cost, self.H, self.train_op]
                    feed_dict = {X1: in_idx, X2: context, Y: out_idx, M: len(iters)}
                    for j in range(len(self.layers)):
                        new_weights[f"H{j}"] = state[j]
                    self.tf_variables.set_weights(new_weights)
                    cost, state, _ = sess.run(fetches, feed_dict)
                    reset = (start + i + 1 == end - 1)
                    if reset.sum() > 0:
                        for i in range(len(self.layers)):
                            state[i][reset] = 0
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                start = start + minlen - 1
                finished_mask = (end - start <= 1)
                n_finished = finished_mask.sum()
                iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
                maxiter += n_finished
                valid_mask = (iters < len(offset_sessions) - 1)
                n_valid = valid_mask.sum()
                if n_valid == 0 or maxiter > len(offset_sessions):
                    finished = True
                    break
                mask = finished_mask & valid_mask
                sessions = session_idx_arr[iters[mask]]
                start[mask] = offset_sessions[sessions]
                end[mask] = offset_sessions[sessions + 1]
                iters = iters[valid_mask]
                start = start[valid_mask]
                end = end[valid_mask]
                for i in range(len(self.layers)):
                    state[i][finished_mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
            end_time = time.time()
            print("cost: {}, time:{}".format(avgc, (end_time - start_time) / 60))
        self.saver.save(sess, '{}/gru4rec-with-context-model'.format(self.checkpoint_dir), global_step=epoch)

    def getWeights(self):
        self.weights = self.tf_variables.get_weights()

    def setWeights(self, new_weights):
        self.tf_variables.set_weights(new_weights)

    def symbolic_predict(self, X1, X2, Y, M, H, items):
        if items is not None:
            H_new, yhat, _, _, _ = self.model(X1, X2, H, M, R=None, Y=Y, predict=True)
        else:
            H_new, yhat, _, _, _ = self.model(X1, X2, H, M, R=None, Y=None, predict=True)
        return yhat, H_new


class Gru4recWithPreEmbedding(GruBase):
    def __init__(self, is_straining, session_key, item_key, time_key, batch_size, embedding=0,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 hidden_act='tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100,
                 n_epochs=10,
                 dropout_p_hidden=0, learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0, dropout_p_embed=0):
        super().__init__(is_straining, session_key, item_key, time_key, batch_size, embedding,
                         initializer,
                         hidden_act, final_act, loss, grad_cap, layers, rnn_size, n_epochs,
                         dropout_p_hidden, learning_rate, checkpoint_dir,
                         adapt, pre_embedding_y, pre_embedding_x, dropout_p_embed)

    def process_data(self, data, embedding_data):
        itemids = np.unique(embedding_data[self.item_key])
        itemids = [x for x in itemids if x in data[self.item_key].unique()]
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        embedding_data = pd.merge(embedding_data,
                                  pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                                  on=self.item_key, how='inner')
        embedding_data.sort_values(["ItemIdx"], inplace=True)
        PE_MATRIX = embedding_data["Embeddings"].values
        PE_MATRIX = np.array(PE_MATRIX.tolist())
        PE_MATRIX = PE_MATRIX.astype(np.float)

        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data.astype({'ItemIdx': 'int32'})
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return data, offset_sessions, PE_MATRIX

    def init(self):
        self.Wx, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []
        if self.embedding:
            with tf.name_scope("gru_embedding"):
                self.E = tf.compat.v1.get_variable("E", shape=[self.n_items, self.embedding], initializer=self.initializer,
                                         dtype=tf.float32)
                n_features = self.embedding
        elif self.pre_embedding_x:
            n_features = self.pre_embedding_x
        else:
            n_features = self.n_items
        with tf.variable_scope("gru_weights"):
            for i in range(len(self.layers)):
                self.Wx.append(tf.compat.v1.get_variable("Wx{}".format(i),
                                                         shape=[self.layers[i - 1] if i > 0 else n_features,
                                                                3 * self.layers[i]], initializer=self.initializer,
                                                         dtype=tf.float32))
                self.Wh.append(tf.compat.v1.get_variable('Wh{}'.format(i), shape=[self.layers[i], self.layers[i]],
                                                         initializer=self.initializer, dtype=tf.float32))
                self.Wrz.append(tf.compat.v1.get_variable("Wrz{}".format(i), shape=[self.layers[i], 2 * self.layers[i]],
                                                          initializer=self.initializer, dtype=tf.float32))
                self.Bh.append(tf.compat.v1.get_variable("Bh{}".format(i), shape=[3 * self.layers[i]],
                                                         initializer=tf.zeros_initializer(), dtype=tf.float32))
                self.H.append(tf.compat.v1.get_variable("H{}".format(i), shape=[self.batch_size, self.layers[i]],
                                                        initializer=tf.zeros_initializer(), dtype=tf.float32))
            if self.pre_embedding_y and i == 0:
                self.Wy = tf.compat.v1.get_variable("Wt", shape=[n_features, 1],
                                                    initializer=tf.zeros_initializer(), dtype=tf.float32)

                self.By = tf.compat.v1.get_variable('By', [1], initializer=tf.constant_initializer(0.0),
                                                    dtype=tf.float32)
            else:
                self.Wy = tf.compat.v1.get_variable("Wt", shape=[self.n_items, self.layers[-1]],
                                                    initializer=tf.zeros_initializer(), dtype=tf.float32)
                self.By = tf.compat.v1.get_variable('By', [self.n_items, 1], initializer=tf.constant_initializer(0.0),
                                                    dtype=tf.float32)
                # self.global_step = tf.compat.v1.get_variable(0, name='global_step', trainable=False)

    def model(self, X, H, M, PE, R=None, Y=None, drop_p_hidden=0.0, drop_p_embed=0.0, predict=False):
        sparams, full_params, sidxs = [], [], []
        if self.embedding:
            Sx = tf.nn.embedding_lookup(self.E, X)
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
            sparams.append(Sx)
            full_params.append(self.E)
            sidxs.append(X)
        elif self.pre_embedding_x:
            Sx = tf.nn.embedding_lookup(PE, X)
            y = Sx
            H_new = []
            start = 0
        else:
            Sx = tf.nn.embedding_lookup(self.Wx[0], X)
            vec = Sx + self.Bh[0]
            rz = tf.nn.sigmoid(vec[:, self.layers[0]:] + tf.matmul(H[0], self.Wrz[0]))
            h = self.hidden_activation(tf.matmul(H[0] * rz[:, :self.layers[0]], self.Wh[0]) + vec[:, :self.layers[0]])
            z = rz[:, self.layers[0]:]
            h = (1.0 - z) * H[0] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            start = 1
            H_new = [h]
            sparams.append(Sx)
            full_params.append(self.Wx[0])
            sidxs.append(X)
        for i in range(start, len(self.layers)):
            vec = tf.matmul(y, self.Wx[i]) + self.Bh[i]
            rz = tf.nn.sigmoid(vec[:, self.layers[i]:] + tf.matmul(H[i], self.Wrz[i]))
            h = self.hidden_activation(tf.matmul(H[i] * rz[:, :self.layers[i]], self.Wh[i]) + vec[:, :self.layers[i]])
            z = rz[:, self.layers[i]:]
            h = (1.0 - z) * H[i] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            H_new.append(h)
        if Y is not None:
            if self.pre_embedding_y:
                aha = tf.transpose(tf.tile(self.Wy, tf.constant([1, PE.shape[0]], tf.int32)) * tf.transpose(PE))
                Sy = tf.nn.embedding_lookup(aha, Y)
            else:
                Sy = tf.nn.embedding_lookup(self.Wy, Y)
                sparams.append(Sy)
                full_params.append(self.Wy)
                sidxs.append(Y)
                SBy = tf.nn.embedding_lookup(self.By, Y)
                sparams.append(SBy)
                full_params.append(self.By)
                sidxs.append(Y)

            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(tf.matmul(y, tf.transpose(Sy.T)) + tf.squeeze(self.By))
            else:
                y = self.final_activation(tf.matmul(y, tf.transpose(Sy)) + tf.squeeze(self.By))
            return H_new, y, sparams, full_params, sidxs
        else:
            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(tf.matmul(y, tf.transpose(self.Wy)) + tf.squeeze(self.self.By))
            else:
                y = self.final_activation(tf.matmul(y, tf.transpose(self.Wy)) + tf.squeeze(self.By))
            return H_new, y, sparams, full_params, sidxs

    def fit(self, data, pe_matrix, offset_sessions, sess):
        self.predict = None
        self.error_during_train = False
        # Setup placeholders
        X = tf.compat.v1.placeholder(tf.int32, [None], name='input')
        Y = tf.compat.v1.placeholder(tf.int32, [None], name='output')
        R = tf.compat.v1.placeholder(tf.bool, [None], name='reset')
        M = tf.compat.v1.placeholder(tf.float32, [], name='m')
        PE = tf.compat.v1.placeholder(tf.float32, [pe_matrix.shape[0], pe_matrix.shape[1]], name='pre_item_embedding')
        # Build tensorflow model
        H_new, Y_pred, sparams, full_params, sidxs = self.model(X, self.H, M, PE, R, Y, self.dropout_p_hidden,
                                                                self.dropout_p_embed)
        params = [self.Wx if self.embedding else self.Wx[1:], self.Wh, self.Wrz, self.Bh]
        # Setup cost, optimizer and train_op
        self.cost = (M / self.batch_size) * self.loss_function(Y_pred)
        tvars = tf.compat.v1.trainable_variables()
        optimizer = RMSPropWithMomentum(lr=0.1, epsilon=1e-06, momentum=0.3)
        self.train_op = optimizer.get_updates(self.cost, tvars)
        # Initialize global variable
        sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)
        # Initialize TensorflowVariablses
        self.tf_variables = TensorflowVariables.TensorflowVariables(self.cost, sess)

        for epoch in range(self.n_epochs):
            start_time = time.time()
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(len(self.layers))]
            epoch_cost = []
            session_idx_arr = np.arange(len(offset_sessions) - 1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            new_weights = {}
            while not finished:
                minlen = (end - start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start + i + 1]
                    fetches = [self.cost, self.H, self.train_op]
                    feed_dict = {X: in_idx, Y: out_idx, M: len(iters), PE: pe_matrix}
                    for j in range(len(self.layers)):
                        new_weights[f"H{j}"] = state[j]
                    self.tf_variables.set_weights(new_weights)
                    cost, state, _ = sess.run(fetches, feed_dict)
                    reset = (start + i + 1 == end - 1)
                    if reset.sum() > 0:
                        for i in range(len(self.layers)):
                            state[i][reset] = 0
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                start = start + minlen - 1
                finished_mask = (end - start <= 1)
                n_finished = finished_mask.sum()
                iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
                maxiter += n_finished
                valid_mask = (iters < len(offset_sessions) - 1)
                n_valid = valid_mask.sum()
                if n_valid == 0 or maxiter > len(offset_sessions):
                    finished = True
                    break
                mask = finished_mask & valid_mask
                sessions = session_idx_arr[iters[mask]]
                start[mask] = offset_sessions[sessions]
                end[mask] = offset_sessions[sessions + 1]
                iters = iters[valid_mask]
                start = start[valid_mask]
                end = end[valid_mask]
                for i in range(len(self.layers)):
                    state[i][finished_mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
            end_time = time.time()
            print("cost: {}, time:{}".format(avgc, (end_time - start_time) / 60))
        self.saver.save(sess, '{}/gru4rec-model'.format(self.checkpoint_dir), global_step=epoch)

    def getWeights(self):
        self.weights = self.tf_variables.get_weights()

    def setWeights(self, new_weights):
        self.tf_variables.set_weights(new_weights)

    def symbolic_predict(self, X, Y, M, H, PE, items):
        if items is not None:
            H_new, yhat, _, _, _ = self.model(X, H, M, PE, R=None, Y=Y, predict=True)
        else:
            H_new, yhat, _, _, _ = self.model(X, H, M, PE, R=None, Y=None, predict=True)
        return yhat, H_new
