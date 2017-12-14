import os
import sys
sys.path.append(os.path.join(
  os.path.abspath(os.path.dirname(__file__)),
   '../utils')
  )

from copy import copy
import argparse

import numpy as np

from config import *

from transition_system import Transition
from conllu.conllu import CONLLU

from word_vec.vector import Vector
V = Vector()

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='train,test', type=str, help='File Path')
parser.add_argument('--input_file', default='', type=str, help='Data Path')
parser.add_argument('--output_file', default='', type=str, help='Output File Path')
parser.add_argument('--max_examples', default=50, type=int, help='Max sentences to parse')
args = parser.parse_args()

def vectorize_features(features):
  data = []
  for inp in features:
    data.append([V.word_vectors(inp['W']),
                 V.pos_vectors(inp['T']),
                 V.arclabel_vectors(inp['L']),
                 ])
  return np.array(data)

def main(input, output, max_examples):
  sentences = []

  print('Log: Opening data file..')

  with open(input, 'r+') as input_file:
    for line in input_file.readlines():
      sentences.append(line)
    content = ''.join(sentences)

  print('Log: Parsing CONLL-U..')

  conllu = CONLLU()
  parsed_conllu = conllu.parse(content)

  print('Log: Parsing transitions..')

  projective = 0
  non_projective = 0
  data = []

  limited_sent = parsed_conllu if max_examples == -1 else parsed_conllu[:max_examples]
  for sentence in limited_sent:
    T = Transition(copy(sentence))

    try:
      features = T.parse().extract_features()
      projective += 1
    except:
      non_projective += 1
    else:
      vectorized = vectorize_features(features[0])
      data.append([ [vectorized[i], features[1][i]] for i in range(len(vectorized)) ])

  print('Log: Projective trees:{p}, Non-projective trees:{n_p}'\
        .format(p=projective, n_p=non_projective))
    
  print('Log: Saving output..')

  np.save(output, np.array(data))
  
  print('Log: Done.')

if __name__ == '__main__':
  if args.input_file and args.output_file:
    main(args.input_file, args.output_file, max_examples=args.max_examples)
  else:
    if args.file:
      if 'train' in args.file:
        main(PATH_CONLLU_TRAIN, PATH_PARSED_DATA_TRAIN, max_examples=args.max_examples)
      if 'test' in args.file:
        main(PATH_CONLLU_TEST, PATH_PARSED_DATA_TEST, max_examples=args.max_examples)
