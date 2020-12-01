# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from services import TensorflowVariables
tf.compat.v1.disable_eager_execution()

def evaluation_gru4rec(test_data, train_data, items, is_items_vs_all, is_out_of_time, cut_off, batch_size,
         session_key, item_key, time_key, gru, gru_config):
    """
    Function evaluates recall and mrr for a subset of items.
    :param gru: It is an object of class Gru4rec
    :param gru_config: It is a dictionary for weights and other configration needed for configuring gru
    :param test_data:
    :param train_data:
    :param items: items of interest
    :param is_items_vs_all: if true, eval ranks items comparing to all valid items. Else eval ranks items comparing to eachother
    :param is_out_of_time: if true, part of users sessions should be in training data and the rest in test data
    :param cut_off:
    :param batch_size:
    :return:
    """
    tf.compat.v1.reset_default_graph()
    # If is_out_of_time is true, combine train_data and test_data and tag test_data as "in-eval"
    if is_out_of_time:
        test_users = test_data[session_key].unique()
        train_data = train_data[train_data[session_key].isin(test_users)].copy()
        train_data['in_eval'] = False
        test_data['in_eval'] = True
        test_data = pd.concat([train_data, test_data])
    else:
        test_data['in_eval'] = True

    # Setup gru graph
    gru.predict = False
    gru.is_training = False
    gru.n_items = gru_config["n_items"]
    gru.itemidmap = gru_config["itemidmap"]
    if items is None:
        items = gru.itemidmap.index
    gru.init()
    X = tf.compat.v1.placeholder(tf.int32, [None], name='X')
    Y = tf.compat.v1.placeholder(tf.int32, [None], name='Y')
    R = tf.compat.v1.placeholder(tf.bool, [None], name='R')
    M = tf.compat.v1.placeholder(tf.float32, [], name='M')
    H = [tf.compat.v1.placeholder(tf.float32, [None, gru.rnn_size], name='State') for _ in range(len(gru.layers))]
    state = [np.zeros([batch_size, gru.rnn_size], dtype=np.float32) for _ in range(len(gru.layers))]
    yhat, H_new = gru.symbolic_predict(X, Y, M, H, items)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf_variables = TensorflowVariables.TensorflowVariables(yhat, sess)
        tf_variables.set_weights(gru_config["weights"])

        # prepare inputs, targeted outputs and hidden states
        fetches = [yhat, H_new]
        test_data = pd.merge(test_data,
                             pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                             on=item_key, how='inner')
        test_data.sort_values([session_key, time_key, item_key], inplace=True)
        test_data_items = test_data.ItemIdx.values
        test_data_itemkey = test_data[item_key].values
        # If is_items_vs all is true, rank items vs all valid items
        if is_items_vs_all:
            item_idxs = gru.itemidmap.values
        # If is_items_vs all is false, rank items vs eachother
        else:
            item_idxs = gru.itemidmap[items]
        recall, mrr, n = 0, 0, 0
        offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
        session_idx_arr = np.arange(len(offset_sessions) - 1)
        iters = np.arange(batch_size)
        start = offset_sessions[session_idx_arr[iters]]
        end = offset_sessions[session_idx_arr[iters] + 1]
        # in_eval_mask = np.zeros(batch_size, dtype=np.bool)
        maxiter = iters.max()
        # start = offset_sessions[iters]
        # end = offset_sessions[iters + 1]
        finished = False
        cidxs = []
        while not finished:
            minlen = (end - start).min()
            out_idx = test_data_items[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = test_data_items[start + i + 1]
                out_itemkey = test_data_itemkey[start + i + 1]
                y = np.hstack([out_idx, item_idxs])
                I = np.in1d(out_itemkey, items)
                J = in_idx != out_idx
                in_eval_mask = test_data['in_eval'].values[start + i + 1]
                feed_dict = {X: in_idx, Y: y, M: len(iters)}
                for j in range(len(gru.layers)):
                    feed_dict[H[j]] = state[j]
                preds, state = sess.run(fetches, feed_dict)
                reset = (start + i + 1 == end - 1)
                if reset.sum() > 0:
                    for i in range(len(gru.layers)):
                        state[i][reset] = 0
                preds = np.asarray(preds)
                targets = np.diag(preds[:, :len(out_idx)])
                others = preds[:, len(out_idx):]
                if np.logical_and(in_eval_mask, I).sum():
                    ranks = (others.T >= targets).sum(axis=0)[np.logical_and(np.logical_and(in_eval_mask, I), J)]
                    ranks[ranks == 0] = 1
                    rec = (ranks <= cut_off).sum()
                    m = ((ranks <= cut_off) / ranks).sum()
                    recall += rec
                    mrr += m
                    n += len(ranks)
                    if (ranks == 0).sum() > 0:
                        print("there exist ranks zeros")
            start = start + minlen - 1
            finished_mask = (end - start <= 1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(offset_sessions) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = session_idx_arr[iters[mask]]
            start[mask] = offset_sessions[sessions]
            end[mask] = offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if valid_mask.any():
                for i in range(len(gru.layers)):
                    state[i] = state[i][valid_mask]
        return recall / n, mrr / n


def evaluation_gru4rec_with_context(test_data, train_data, context_columns, n_context_columns, items, is_items_vs_all, is_out_of_time, cut_off, batch_size,
         session_key, item_key, time_key, gru, gru_config):
    """
    Function evaluates recall and mrr for a subset of items.
    :param gru: It is an object of class Gru4recWithContext
    :param gru_config: It is a dictionary for weights and other configration needed for configuring gru
    :param test_data:
    :param train_data:
    :param items: items of interest
    :param is_items_vs_all: if true, eval ranks items comparing to all valid items. Else eval ranks items comparing to eachother
    :param is_out_of_time: if true, part of users sessions should be in training data and the rest in test data
    :param cut_off:
    :param batch_size:
    :return:
    """
    tf.compat.v1.reset_default_graph()
    # If is_out_of_time is true, combine train_data and test_data and tag test_data as "in-eval"
    if is_out_of_time:
        test_users = test_data[session_key].unique()
        train_data = train_data[train_data[session_key].isin(test_users)].copy()
        train_data['in_eval'] = False
        test_data['in_eval'] = True
        test_data = pd.concat([train_data, test_data])
    else:
        test_data['in_eval'] = True

    # Get context
    data_context = np.array(test_data[context_columns].tolist())
    # Setup gru graph
    gru.predict = False
    gru.is_training = False
    gru.n_items = gru_config["n_items"]
    gru.itemidmap = gru_config["itemidmap"]
    if items is None:
        items = gru.itemidmap.index
    gru.init(n_context_columns)
    X1 = tf.compat.v1.placeholder(tf.int32, [None], name='X1')
    X2 = tf.compat.v1.placeholder(tf.float32, [None, None], name='X2')
    Y = tf.compat.v1.placeholder(tf.int32, [None], name='Y')
    R = tf.compat.v1.placeholder(tf.bool, [None], name='Reset')
    M = tf.compat.v1.placeholder(tf.float32, [], name='M')
    H = [tf.compat.v1.placeholder(tf.float32, [None, gru.rnn_size], name='State') for _ in range(len(gru.layers))]
    state = [np.zeros([batch_size, gru.rnn_size], dtype=np.float32) for _ in range(len(gru.layers))]
    yhat, H_new = gru.symbolic_predict(X1, X2, Y, M, H, items)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf_variables = TensorflowVariables.TensorflowVariables(yhat, sess)
        tf_variables.set_weights(gru_config["weights"])

        # prepare inputs, targeted outputs and hidden states
        fetches = [yhat, H_new]
        test_data = pd.merge(test_data,
                             pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                             on=item_key, how='inner')
        test_data.sort_values([session_key, time_key, item_key], inplace=True)
        test_data_items = test_data.ItemIdx.values
        test_data_itemkey = test_data[item_key].values
        # If is_items_vs all is true, rank items vs all valid items
        if is_items_vs_all:
            item_idxs = gru.itemidmap.values
        # If is_items_vs all is false, rank items vs eachother
        else:
            item_idxs = gru.itemidmap[items]
        recall, mrr, n = 0, 0, 0
        offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
        session_idx_arr = np.arange(len(offset_sessions) - 1)
        iters = np.arange(batch_size)
        start = offset_sessions[session_idx_arr[iters]]
        end = offset_sessions[session_idx_arr[iters] + 1]
        # in_eval_mask = np.zeros(batch_size, dtype=np.bool)
        maxiter = iters.max()
        # start = offset_sessions[iters]
        # end = offset_sessions[iters + 1]
        finished = False
        cidxs = []
        while not finished:
            minlen = (end - start).min()
            out_idx = test_data_items[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                context = data_context[start + i]
                out_idx = test_data_items[start + i + 1]
                out_itemkey = test_data_itemkey[start + i + 1]
                y = np.hstack([out_idx, item_idxs])
                I = np.in1d(out_itemkey, items)
                J = in_idx != out_idx
                in_eval_mask = test_data['in_eval'].values[start + i + 1]
                feed_dict = {X1: in_idx, X2: context, Y: y, M: len(iters)}
                for j in range(len(gru.layers)):
                    feed_dict[H[j]] = state[j]
                preds, state = sess.run(fetches, feed_dict)
                reset = (start + i + 1 == end - 1)
                if reset.sum() > 0:
                    for i in range(len(gru.layers)):
                        state[i][reset] = 0
                preds = np.asarray(preds)
                targets = np.diag(preds[:, :len(out_idx)])
                others = preds[:, len(out_idx):]
                if np.logical_and(in_eval_mask, I).sum():
                    preds_for_targest = preds[[np.logical_and(np.logical_and(in_eval_mask, I), J)]]
                    # preds_for_targest.apply(lambda x: np.std(x), axis=1)
                    ranks = (others.T >= targets).sum(axis=0)[np.logical_and(np.logical_and(in_eval_mask, I), J)]
                    ranks[ranks == 0] = 1
                    rec = (ranks <= cut_off).sum()
                    m = ((ranks <= cut_off) / ranks).sum()
                    recall += rec
                    mrr += m
                    n += len(ranks)
                    if (ranks == 0).sum() > 0:
                        print("there exist ranks zeros")
            start = start + minlen - 1
            finished_mask = (end - start <= 1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(offset_sessions) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = session_idx_arr[iters[mask]]
            start[mask] = offset_sessions[sessions]
            end[mask] = offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if valid_mask.any():
                for i in range(len(gru.layers)):
                    state[i] = state[i][valid_mask]
        return recall / n, mrr / n


def evaluation_gru4rec_pre_embeddin(event_dict, test_data, train_data, embedding_data, items, is_items_vs_all, is_out_of_time, cut_off, batch_size,
         session_key, item_key, time_key, gru, gru_config):
    """
    Function evaluates recall and mrr for a subset of items.
    :param gru: It is an object of class Gru4recWithContext
    :param gru_config: It is a dictionary for weights and other configration needed for configuring gru
    :param test_data:
    :param train_data:
    :param items: items of interest
    :param is_items_vs_all: if true, eval ranks items comparing to all valid items. Else eval ranks items comparing to eachother
    :param is_out_of_time: if true, part of users sessions should be in training data and the rest in test data
    :param cut_off:
    :param batch_size:
    :return:
    """
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    # If is_out_of_time is true, combine train_data and test_data and tag test_data as "in-eval"
    if is_out_of_time:
        test_users = test_data[session_key].unique()
        train_data = train_data[train_data[session_key].isin(test_users)].copy()
        train_data['in_eval'] = False
        test_data['in_eval'] = True
        test_data = pd.concat([train_data, test_data])
    else:
        test_data['in_eval'] = True
    test_data.loc[:, "row_number"] = [i for i in range(test_data.shape[0])]
    event_dict_reverse = {event_dict[x]: x for x in event_dict}
    test_data.loc[:, "ContentId"] = [event_dict_reverse[str(int(x))] for x in test_data["ItemId"]]

    embedding_data.loc[:, "ItemEmbId"] = [i for i in range(embedding_data.shape[0])]
    embedding_data.sort_values(["ItemEmbId"], inplace=True)
    PE_MATRIX = embedding_data["Embeddings"].values
    PE_MATRIX = np.array(PE_MATRIX.tolist())
    PE_MATRIX = PE_MATRIX.astype(np.float)
    test_data = test_data.merge(embedding_data, on="ContentId", how='inner')
    test_data.sort_values(["SessionId", "Time"], inplace=True)
    test_data.reset_index(inplace=True)
    item_idxs = test_data["ItemEmbId"].unique()
    test_data_itemkey = test_data["ContentId"].values
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()

    # Setup gru graph
    gru.predict = False
    gru.is_training = False
    gru.n_items = gru_config["n_items"]
    gru.itemidmap = gru_config["itemidmap"]
    if items is None:
        items = test_data["ContentId"].unique()#gru.itemidmap.index
    gru.init()
    X = tf.compat.v1.placeholder(tf.int32, [None], name='input')
    Y = tf.compat.v1.placeholder(tf.int32, [None], name='output')
    R = tf.compat.v1.placeholder(tf.bool, [None], name='reset')
    M = tf.compat.v1.placeholder(tf.float32, [], name='m')
    PE = tf.compat.v1.placeholder(tf.float32, [PE_MATRIX.shape[0], PE_MATRIX.shape[1]], name='pre_item_embedding')
    H = [tf.compat.v1.placeholder(tf.float32, [None, gru.rnn_size], name='State') for _ in range(len(gru.layers))]
    state = [np.zeros([batch_size, gru.rnn_size], dtype=np.float32) for _ in range(len(gru.layers))]
    yhat, H_new = gru.symbolic_predict(X, Y, M, H, PE, items)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf_variables = TensorflowVariables.TensorflowVariables(yhat, sess)
        tf_variables.set_weights(gru_config["weights"])

        # prepare inputs, targeted outputs and hidden states
        fetches = [yhat, H_new]
        recall, mrr, n = 0, 0, 0
        session_idx_arr = np.arange(len(offset_sessions) - 1)
        iters = np.arange(batch_size)
        start = offset_sessions[session_idx_arr[iters]]
        end = offset_sessions[session_idx_arr[iters] + 1]
        maxiter = iters.max()
        finished = False
        cidxs = []
        while not finished:
            minlen = (end - start).min()
            out_idx = test_data["ItemEmbId"].values[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = test_data["ItemEmbId"].values[start + i + 1]
                out_itemkey = test_data_itemkey[start + i + 1]
                y = np.hstack([out_idx, item_idxs])
                I = np.in1d(out_itemkey, items)
                J = in_idx != out_idx
                in_eval_mask = test_data['in_eval'].values[start + i + 1]
                feed_dict = {X: in_idx, PE: PE_MATRIX, Y: y, M: len(iters)}
                for j in range(len(gru.layers)):
                    feed_dict[H[j]] = state[j]
                preds, state = sess.run(fetches, feed_dict)
                reset = (start + i + 1 == end - 1)
                if reset.sum() > 0:
                    for i in range(len(gru.layers)):
                        state[i][reset] = 0
                preds = np.asarray(preds)
                targets = np.diag(preds[:, :len(out_idx)])
                others = preds[:, len(out_idx):]
                if np.logical_and(in_eval_mask, I).sum():
                    ranks = (others.T >= targets).sum(axis=0)[np.logical_and(np.logical_and(in_eval_mask, I), J)]
                    ranks[ranks == 0] = 1
                    rec = (ranks <= cut_off).sum()
                    m = ((ranks <= cut_off) / ranks).sum()
                    recall += rec
                    mrr += m
                    n += len(ranks)
                    if (ranks == 0).sum() > 0:
                        print("there exist ranks zeros")
            start = start + minlen - 1
            finished_mask = (end - start <= 1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = (iters < len(offset_sessions) - 1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = session_idx_arr[iters[mask]]
            start[mask] = offset_sessions[sessions]
            end[mask] = offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if valid_mask.any():
                for i in range(len(gru.layers)):
                    state[i] = state[i][valid_mask]
        return recall / n, mrr / n