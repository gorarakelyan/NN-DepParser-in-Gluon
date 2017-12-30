import argparse

import mxnet as mx

from config import *

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=NN_BATCH_SIZE, type=int, help='Batch Size')
parser.add_argument('--hidden_units', default=NN_HIDDEN_UNITS, type=int, help='Hidden Units')
parser.add_argument('--drop_out', default=NN_DROP_OUT, type=float, help='Dropout rate')

parser.add_argument('--test_data', default=PATH_PARSED_DATA_TEST, type=str, help='Test Data Path')
parser.add_argument('--ctx', default=DEFAULT_CTX, type=str, help='Context')
args = parser.parse_args()

net = MLP(drop_out=args.drop_out, 
          hidden_units=args.hidden_units)

net.set_ctx(args.ctx)

net.load_model()

data = net.prepare_data(path=args.test_data,
                        batch_size=args.batch_size,
                        train=False)

accuracy = net.evaluation(data)
print('Log: Test Accuracy - {acc}'.format(acc=accuracy))
