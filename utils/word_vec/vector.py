import os
import sys
sys.path.append(os.path.join(
  os.path.abspath(os.path.dirname(__file__)),
   '../parser/config')
  )

import numpy as np
import word2vec as wv

from config import *

ROOT = PATH_DATA_WORD2VEC
BIN_PATH = ''.join([ROOT, 'text.bin'])

class Vector(object):
  model = None
  pos_model = None
  arclabel_model = None

  def __init__(self):
    if os.path.isfile(BIN_PATH):
      self.model = wv.load(BIN_PATH)

    self.pos_model = {i: np.random.uniform(-1, 1, size=(VEC_SIZE,)) for i in POS_LABELS}
    self.pos_model.update({'NULL': np.zeros(VEC_SIZE)})

    self.arclabel_model = {i: np.random.uniform(-1, 1, size=(VEC_SIZE,)) for i in ARC_LABELS}
    self.arclabel_model.update({'NULL': np.zeros(VEC_SIZE)})

  def vector(self, word):
    try:
      return self.model[word]
    except:
      try:
        return self.model[word.lower()]
      except:
        return np.zeros(VEC_SIZE)

  def word_vectors(self, arr):
    vectors = []
    for w in arr:
      vectors.append(self.vector(w))
    return np.array(vectors)

  def pos_vectors(self, arr):
    vectors = []
    for t in arr:
      vectors.append(self.pos_model[t])
    return np.array(vectors)

  def arclabel_vectors(self, arr):
    vectors = []
    for l in arr:
      vectors.append(self.arclabel_model[l])
    return np.array(vectors)