import sys
sys.path.append('./')
from services import Gru4rec, evaluation
from constant import data_key
import _pickle as cPickle
import os
import tensorflow as tf

import pandas as pd
import argparse

import datetime
from datetime import datetime, timedelta




def main(args):
    local_data_dir = args.local_data_dir
    model_dir = args.model_dir
    train_file_name = args.train_file_name
    batch_size = args.batch_size

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
    data.dropna(inplace=True)
    data[data_key.ITEM_KEY] = pd.to_numeric(data[data_key.ITEM_KEY])
    test = data[data[data_key.TIME_KEY] >= (max(data[data_key.TIME_KEY]) - 86400)]
    data = data[data[data_key.TIME_KEY] < (max(data[data_key.TIME_KEY]) - 86400)]
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

    gru_config = cPickle.load(open(model_dir+"/gru4rec_config", "rb"))
    for cut_off in [10, 20]:
        res = evaluation.evaluation_gru4rec_with_context(test_data=test, train_data=data, context_columns="Context", n_context_columns=n_context_column,items=None, cut_off=cut_off, is_items_vs_all=True, is_out_of_time=False,
                        session_key=data_key.SESSION_KEY, item_key=data_key.ITEM_KEY, time_key=data_key.TIME_KEY,
                        gru=gru, gru_config=gru_config, batch_size=int(batch_size))

        print('Recall@{}: {}'.format(cut_off, res[0]))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('local_data_dir', type=str, help='Data directory', default="./gru/data/{}")
    parser.add_argument('model_dir', type=str, help='Model checkpoint directory', default="./gru/checkpoints/")
    parser.add_argument("train_file_name", type=str, help="train data file name", default="sample_data.csv")
    parser.add_argument("batch_size", type=int, help="Batch size", default="400")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))