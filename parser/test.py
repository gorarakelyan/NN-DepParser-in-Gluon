import argparse

import mxnet as mx

from config import *

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=NN_BATCH_SIZE, type=int, help='Batch Size')
parser.add_argument('--hidden_units', default=NN_HIDDEN_UNITS, type=int, help='Hidden Units')
parser.add_argument('--drop_out', default=NN_DROP_OUT, type=float, help='Dropout rate')

parser.add_argument('--test_data', default=PATH_PARSED_DATA_TEST, type=str, help='Test Data Path')
parser.add_argument('--dataset_size', default=DEFAULT_DATASET_SIZE, type=int, help='Train Dataset size')
parser.add_argument('--ctx', default=DEFAULT_CTX, type=str, help='Context')
args = parser.parse_args()

net = MLP(drop_out=args.drop_out, 
          hidden_units=args.hidden_units)

net.set_ctx(args.ctx)

net.load_model()

data_attr = {'path': args.test_data,
             'dataset_size': args.dataset_size,
             'batch_size': args.batch_size,
             'shuffle_data': False, 
             }

cumulative_accuracy = 0
set_count = 0
data_gen = net.prepare_data(**data_attr)

for test_data in data_gen:
  set_count += 1
  set_acc = net.evaluation(test_data)
  cumulative_accuracy += set_acc
  print('Log: Dataset Accuracy - {acc}'.format(acc=set_acc))

accuracy = cumulative_accuracy / set_count
print('Log: Test Accuracy - {acc}'.format(acc=accuracy))