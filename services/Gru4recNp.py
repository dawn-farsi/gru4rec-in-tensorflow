import numpy as np
import pandas as pd
import time


class GruBaseNp(object):
    def __init__(self,is_straining, session_key, item_key, time_key, batch_size, embedding = 0 ,
                 hidden_act = 'tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100, n_epochs=10,
                 learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0, ):
        """
        :param session_key: string, header of the session ID column in the input file (default: 'SessionId')
        :param item_key: string, header of the item ID column in the input file (default: 'ItemId')
        :param time_key: string
        :param layers: list of int values, list of the number of GRU units in the layers (default : [100])
        :param batch_size : int, size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 32)
        :param pre_embedding_y: int, size of the embedding used for output
        :param pre_embedding_x: int, size of the embedding used for input
        :param embedding : int, size of the embedding used, 0 means not to use embedding (default: 0)
        :param adapt: string, sets the appropriate learning rate adaptation strategy, (default: "adagrad")
        :param hidden_act: string, 'linear', 'relu', 'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function on the hidden states, <X> and <Y> are the parameters of the activation function (default : 'tanh')
        :param final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>' selects the activation function of the final layer, <X> and <Y> are the parameters of the activation function (default : 'elu-1')
        """
        self.is_training = is_straining
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.batch_size = batch_size
        self.embedding = embedding
        self.pre_embedding_x = pre_embedding_x
        self.pre_embedding_y = pre_embedding_y

        self.hidden_act = hidden_act
        self.set_hidden_activation(self.hidden_act)
        self.final_act = final_act
        self.set_final_activation(self.final_act)
        self.loss = loss
        self.grad_cap = grad_cap
        self.layers = layers
        self.rnn_size=rnn_size
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        self.checkpoint_dir=checkpoint_dir
        self.n_items = None
        self.itemidmap = None

    def linear(self, inp):
        return inp

    def tanh(self, inp):
        return np.tanh(inp)

    def relu(self, inp):
        inp[inp < 0] = 0
        return inp

    def sigmoid(self, inp):
        return 1/(1 + np.exp(-inp))

    def softmax(self, inp):
        return np.exp(inp) / sum(np.exp(inp))

    class Elu:
        def __init__(self, alpha):
            self.alpha = alpha
            self.elu = lambda Z: np.where(Z > 0, Z, self.alpha * (np.exp(Z) - 1))

        def execute(self, inp):
            return self.elu(inp)

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
        elif final_act == 'tanh':
            self.final_activation = self.tanh
        elif final_act.startswith('elu-'):
            self.final_activation = self.Elu(float(final_act.split('-')[1])).execute
        else:
            raise NotImplementedError


class Gru4recNp(GruBaseNp):
    def __init__(self,is_straining, session_key, item_key, time_key, batch_size, embedding = 0 ,
                 hidden_act = 'tanh', final_act='elu-0.5', loss="bpr", grad_cap=0, layers=[100], rnn_size=100, n_epochs=10,
                 learning_rate=0.1, checkpoint_dir="",
                 adapt="adagrad", pre_embedding_y=0, pre_embedding_x=0):
        super().__init__(is_straining, session_key, item_key, time_key, batch_size, embedding,
                 hidden_act, final_act, loss, grad_cap, layers, rnn_size, n_epochs,
                 learning_rate, checkpoint_dir,
                 adapt, pre_embedding_y, pre_embedding_x)

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

    def init(self, weights):
        self.Wx = weights["Wx"]
        self.Wh = weights["Wh"]
        self.Wrz = weights["Wrz"]
        self.Bh = weights["Bh"]
        if self.embedding:
            self.E = weights["E"]
        self.Wy=weights["Wy"]
        self.By=weights["By"]


    def model(self, X, H, M, R=None, Y=None, predict=False):
        if self.embedding:
            Sx = self.E[X]
            y = Sx
            H_new = []
            start = 0

        else:
            Sx =self.Wx[0][X]
            vec = Sx + self.Bh[0]
            rz = self.sigmoid(vec[:, self.layers[0]:] + np.matmul(H[0], self.Wrz[0]))
            h = self.hidden_activation(np.matmul(H[0] * rz[:, :self.layers[0]], self.Wh[0]) + vec[:, :self.layers[0]])
            z = rz[:, self.layers[0]:]
            h = (1.0 - z) * H[0] + z * h
            y = h
            start = 1
            H_new = [h]

        for i in range(start, len(self.layers)):
            vec = np.matmul(y, self.Wx[i]) + self.Bh[i]
            rz = np.sigmoid(vec[:, self.layers[i]:] + np.matmul(H[i], self.Wrz[i]))
            h = self.hidden_activation(np.matmul(H[i] * rz[:, :self.layers[i]], self.Wh[i]) + vec[:, :self.layers[i]])
            z = rz[:, self.layers[i]:]
            h = (1.0 - z) * H[i] + z * h
            y = h
            H_new.append(h)
        if Y is not None:
            Sy = self.Wy[Y]
            SBy = self.By[Y]
            y = self.final_activation(np.matmul(y, np.transpose(Sy)) + np.squeeze(SBy))
            return H_new, y
        else:
            y = self.final_activation(np.matmul(y, np.transpose(self.Wy)) + np.squeeze(self.By))
            return H_new, y

    def predict(self, X, Y, M, H, items):
        if items is not None:
            H_new, yhat= self.model(X, H, M, R=None, Y=Y, predict=True)
        else:
            H_new, yhat = self.model(X, H, M, R=None, Y=None, predict=True)
        return yhat, H_new
