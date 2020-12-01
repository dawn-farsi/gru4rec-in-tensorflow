import sys
sys.path.append('./')
from services import Gru4rec
from constant import data_key
import _pickle as cPickle
import os
import tensorflow as tf
from tensorflow.python.framework import ops

import pandas as pd
import argparse
import random

import datetime
from datetime import datetime, timedelta


def main(args):
    local_data_dir = args.local_data_dir
    model_dir = args.model_dir
    train_file_name = args.train_file_name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def add_target_date(unix_time):
        utc_datetime = str(datetime.utcfromtimestamp(unix_time))
        gmt2_time = datetime.strptime(str(utc_datetime)[:19], "%Y-%m-%d %H:%M:%S") + timedelta(hours=2)
        # gmt2_date = str(gmt2_time)
        return gmt2_time

    time_interval = [[6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20], [20, 22], [22, 24]]
    weekdays = [0, 1, 2, 3, 4, 5, 6]

    # reading data locally
    data = pd.read_csv(local_data_dir.format(train_file_name), sep=',')
    data.dropna(inplace = True)
    data[data_key.ITEM_KEY] = pd.to_numeric(data[data_key.ITEM_KEY])
    test = data[data[data_key.TIME_KEY] >= (max(data[data_key.TIME_KEY]) - 86400)]
    data = data[data[data_key.TIME_KEY]<(max(data[data_key.TIME_KEY])-86400)]
    gmt2 = [add_target_date(int(x)) for x in data["Time"]]
    data["WeekDay"] = [x.weekday() for x in gmt2]
    data["TimeInterval"] = [x.hour for x in gmt2]

    data["WeekDay"] = [[1 if x.weekday() == d else 0 for d in weekdays] for x in gmt2]
    data["TimeInterval"] = [[1 if x.hour < int_[1] and x.hour >= int_[0] else 0 for int_ in time_interval] for x in
                            gmt2]
    data["Context"] = [WeekDay + TimeInterval for WeekDay, TimeInterval in zip(data["WeekDay"], data["TimeInterval"])]
    n_context_column = len(data["Context"].iloc[0])

    test = data[data[data_key.TIME_KEY] >= (max(data[data_key.TIME_KEY]) - 86400)]
    train = data[data[data_key.TIME_KEY] < (max(data[data_key.TIME_KEY]) - 86400)]
    ops.reset_default_graph()
    with tf.compat.v1.Session()  as sess:
        gru = Gru4rec.Gru4recWithContext(is_straining=True,
                              session_key=data_key.SESSION_KEY,
                              item_key=data_key.ITEM_KEY,
                              time_key=data_key.TIME_KEY,
                              batch_size=19,
                              embedding=50,
                              initializer=tf.random_normal_initializer(stddev=0.1),
                              hidden_act='tanh',
                              final_act='elu-0.5',
                              loss="bpr",
                              grad_cap=0,
                              layers=[50],
                              rnn_size=50,
                              n_epochs=5,
                              dropout_p_hidden=0,
                              learning_rate=0.1,
                              checkpoint_dir=model_dir)
        train, offset_sessions = gru.process_data(train)
        gru.init(n_context_column)
        gru.fit(train, "Context", n_context_column, offset_sessions, sess)
        gru.getWeights()
        gru_config = {"weights": gru.weights, "n_items":gru.n_items, "itemidmap": gru.itemidmap}
        f = open(model_dir + "/gru4rec_with_context_config", 'wb')
        cPickle.dump(gru_config, f)
        sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('local_data_dir', type=str, help='Data directory', default="./gru/data/{}")
    parser.add_argument('model_dir', type=str, help='Model checkpoint directory', default="./gru/checkpoints/")
    parser.add_argument("train_file_name", type=str, help="train data file name", default="sample_data.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

