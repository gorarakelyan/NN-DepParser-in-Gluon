import os
import sys
sys.path.append(os.path.join(
  os.path.abspath(os.path.dirname(__file__)),
   '../utils')
  )

import argparse

from config import *

from conllu.conllu import CONLLU

parser = argparse.ArgumentParser()
parser.add_argument('--original_file', default=PATH_PREDICT_SENTENCE, type=str, help='Original Data Path')
parser.add_argument('--prediction_file', default=PATH_PREDICT_OUTPUT, type=str, help='Prediction File Path')
parser.add_argument('--validation', default=DEFAULT_VALID, type=str, help='Validation type')
args = parser.parse_args()

def uas(original, prediction):
  if len(original) != len(prediction):
    raise Exception('Error. Original and predicted file lengths are different.')

  length = 0
  correct = 0
  for s in range(len(original)):
    for i in range(len(original[s])):
      length += 1
      if original[s][i]['head'] == prediction[s][i]['head']:
        correct += 1

  return correct / length

def las(original, prediction):
  if len(original) != len(prediction):
    raise Exception('Error. Original and predicted file lengths are different.')

  length = 0
  correct = 0
  for s in range(len(original)):
    for i in range(len(original[s])):
      length += 1
      if original[s][i]['head'] == prediction[s][i]['head'] and \
         original[s][i]['deprel'] == prediction[s][i]['deprel']:
        correct += 1

  return correct / length

def main(origin_path, prediction_path, v_type):
  print('Log: Opening data files..')

  sentences = []
  with open(origin_path, 'r+') as original_file:
    for line in original_file.readlines():
      sentences.append(line)
    content_original = ''.join(sentences)

  sentences = []
  with open(prediction_path, 'r+') as prediction_file:
    for line in prediction_file.readlines():
      sentences.append(line)
    content_pr = ''.join(sentences)

  print('Log: Parsing CONLL-U..')

  conllu = CONLLU()
  parsed_or = conllu.parse(content_original)

  conllu = CONLLU()
  parsed_pr = conllu.parse(content_pr)

  print('Log: Validation..')

  if v_type == 'uas':
    res = uas(parsed_or, parsed_pr)
    return res
  elif v_type == 'las':
    res = las(parsed_or, parsed_pr)
    return res
  else:
    raise Exception('Error. Invalid validation type')


if __name__ == '__main__':
  try:
    accuracy = main(args.original_file, args.prediction_file, v_type=args.validation)
  except Exception as E:
    print(E)
  else:
    print(accuracy)
