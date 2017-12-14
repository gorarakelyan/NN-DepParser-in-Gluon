import os
import sys
sys.path.append(os.path.join(
  os.path.abspath(os.path.dirname(__file__)),
   '../parser/config')
  )

import argparse

import word2vec as wv

from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default=PATH_DATA_WORD2VEC, type=str, help='Train Data Path')
args = parser.parse_args()

ROOT = args.train_data

VOCAB_PATH = ''.join([ROOT, 'text'])
PHRASE_PATH = ''.join([ROOT, 'text-phrases'])
BIN_PATH = ''.join([ROOT, 'text.bin'])
CLUSTER_PATH = ''.join([ROOT, 'text-clusters.txt'])

def train():
  #wv.word2phrase(VOCAB_PATH, PHRASE_PATH, verbose=True)
  wv.word2vec(VOCAB_PATH, BIN_PATH, size=VEC_SIZE, verbose=True)
  wv.word2clusters(VOCAB_PATH, CLUSTER_PATH, VEC_SIZE, verbose=True)

if __name__ == '__main__':
  train()