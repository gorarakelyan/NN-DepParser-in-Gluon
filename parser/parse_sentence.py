import os
import sys
sys.path.append(os.path.join(
  os.path.abspath(os.path.dirname(__file__)),
   '../utils')
  )

import argparse

import mxnet as mx
import numpy as np

from config import *

from model import MLP

from transition_system import Transition
from data import vectorize_features
from conllu.conllu import CONLLU

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_units', default=NN_HIDDEN_UNITS, type=int, help='Hidden Units')
parser.add_argument('--drop_out', default=NN_DROP_OUT, type=float, help='Dropout rate')

parser.add_argument('--ctx', default='cpu', type=str, help='Context')

parser.add_argument('--input_file', default=PATH_PREDICT_SENTENCE, type=str, help='Data Path')
parser.add_argument('--output_file', default=PATH_PREDICT_OUTPUT, type=str, help='Output Path')
args = parser.parse_args()

net = MLP(drop_out=args.drop_out, 
        hidden_units=args.hidden_units)

net.set_ctx(args.ctx)

net.load_model()

sentences = []
with open(args.input_file, 'r+') as input_file:
  for line in input_file.readlines():
    sentences.append(line)
  content = ''.join(sentences)

print('Log: Parsing CONLL-U..')

conllu = CONLLU()
parsed_conllu = conllu.parse(content)
sentence = parsed_conllu[0]

print('Log: Parsing transitions..')

T = Transition(sentence)

invalid = False

while T.next():
  features = T.get_config()
  vectorized = vectorize_features([features])
  input = np.concatenate(tuple((np.array(c).flatten() for c in vectorized[0])))
  
  Y = net.predict(mx.nd.array([input]))
  output = Y[0].asnumpy()
 
  valid = T.get_valid_actions()
  
  for prediction in np.argsort(output, kind='quicksort')[::-1]:
    if prediction in valid:
      print('Log: Action {a} - {p}'.format(a=prediction, p=output[prediction]))
      break

  try:
    if prediction == T.SHIFT_INDEX:
      T.set_action('shift')
    elif prediction < T.RIGHT_ARC_INDEX:
      T.set_action('left_arc:' + ARC_LABELS[prediction])
    else:
      T.set_action('right_arc:' + ARC_LABELS[prediction - T.RIGHT_ARC_INDEX])
  except Exception as E:
    print('Log: Error - {}'.format(E))
    print('Log: Invalid prediction!')
    invalid = True
    break

if not invalid:
  sentence = CONLLU.build(T.get_arcs(root=True), parse_str=True)

  with open(args.output_file, 'w+') as output_file:
    output_file.write(sentence)

  print('Log: Saved!')
