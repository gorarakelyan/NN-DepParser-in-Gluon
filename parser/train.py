import argparse

import mxnet as mx

from config import *

from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=NN_EPOCHS, type=int, help='Epochs')
parser.add_argument('--batch_size', default=NN_BATCH_SIZE, type=int, help='Batch Size')
parser.add_argument('--learning_rate', default=NN_LEARNING_RATE, type=float, help='Learning Rate')
parser.add_argument('--hidden_units', default=NN_HIDDEN_UNITS, type=int, help='Hidden Units')
parser.add_argument('--drop_out', default=NN_DROP_OUT, type=float, help='Dropout rate')

parser.add_argument('--train_data', default=PATH_PARSED_DATA_TRAIN, type=str, help='Train Data Path')
parser.add_argument('--dataset_size', default=DEFAULT_DATASET_SIZE, type=int, help='Train Dataset size')
parser.add_argument('--ctx', default=DEFAULT_CTX, type=str, help='Context')
args = parser.parse_args()

net = MLP(drop_out=args.drop_out, 
          hidden_units=args.hidden_units)

net.set_ctx(args.ctx)

data = {'path': args.train_data,
        'dataset_size': args.dataset_size,
        'batch_size': args.batch_size,
        'shuffle_data': True, 
        }

net.train(data_attr=data,
          epochs=args.epochs,
          learning_rate=args.learning_rate)