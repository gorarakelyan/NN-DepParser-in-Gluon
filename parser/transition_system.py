import re

from copy import copy
from collections import OrderedDict

from config import *

class Transition:
  LEFT_ARC_INDEX = 0
  RIGHT_ARC_INDEX = len(ARC_LABELS)
  SHIFT_INDEX = len(ARC_LABELS) * 2

  def __init__(self, sentence):
    self.sentence = sentence
    self.parsed = None

    self.config = OrderedDict([
      ('stack', []),
      ('buffer', sentence),
      ('arcs', []),
    ])

  def next(self):
    stack = self.config['stack']
    buffer = self.config['buffer']
    if len(buffer) or len(stack) == 0 or len(stack) > 1 or \
      (len(stack) == 1 and len(buffer) != 0):
      return True
    return False
  
  def shift(self):
    #print('S')
    #print(len(self.config['stack']), len(self.config['buffer']))
    item = self.config['buffer'][0]
    del self.config['buffer'][0]
    self.config['stack'].append(item)

    return self

  def right_arc(self, label):
    #print('R')
    stack = self.config['stack']
    item = (stack[-2], stack[-1], label)
    self.config['arcs'].append(item)
    del stack[-1]

    return self

  def left_arc(self, label):
    #print('L')
    stack = self.config['stack']
    item = (stack[-1], stack[-2], label)
    self.config['arcs'].append(item)
    del stack[-2]

    return self

  def buffer_contains_head(self):
    if len(self.config['buffer']) == 0:
      return False

    for node in self.config['buffer']:
      if node['head']== self.config['stack'][-1]['id']:
        return True

    return False

  def get_config(self):
    return self.features(self.config)

  def set_action(self, action):
    if not action:
      return

    if self.next():
      if action == 'shift':
        self.shift()
      else:
        label = action.split(':')[-1]
        if action.split(':')[0] == 'left_arc':
          self.left_arc(label)
        else:
          self.right_arc(label)

  def get_arcs(self):
    return self.config['arcs']

  def parse(self):
    trans = []
    while self.next():
      config = self.config
      curr_config = self.config
      
      if len(config['stack']) < 2:
        action = self.SHIFT_INDEX
        self.shift()
      else:
        if config['stack'][-1]['id'] == config['stack'][-2]['head']:
          label = config['stack'][-2]['deprel']
          action = self.LEFT_ARC_INDEX + ARC_LABELS.index(label)
          self.left_arc(label)
        elif config['stack'][-1]['head'] == config['stack'][-2]['id'] and \
             not self.buffer_contains_head():
          label = config['stack'][-1]['deprel']
          action = self.RIGHT_ARC_INDEX + ARC_LABELS.index(label)
          self.right_arc(label)
        else:
          action = self.SHIFT_INDEX
          self.shift()

      trans.append(({'stack': copy(curr_config['stack']),
                     'buffer': copy(curr_config['buffer']),
                     'arcs': curr_config['arcs'],
                     }, action))

    self.parsed = trans
    return self

  def get_valid_actions(self):
    config = self.config
    if len(config['stack']) > 2:
      valid = list(range(len(ARC_LABELS)*2))
    else:
      valid = []
    if config['buffer']:
      valid.append(len(ARC_LABELS)*2)
    return valid

  def extract_features(self):
    if not self.parsed:
      return

    features = []
    actions = []
    for t in self.parsed:
      features.append(self.features(t[0]))
      actions.append(t[1])

    return [features, actions]

  @classmethod
  def features(cls, config):
    stack = config['stack']
    buffer = config['buffer']
    arcs = config['arcs']

    s1 = cls.node(stack, -1)
    s2 = cls.node(stack, -2)
    s3 = cls.node(stack, -3)
    b1 = cls.node(buffer, -1)
    b2 = cls.node(buffer, -2)
    b3 = cls.node(buffer, -3)

    words = [s1, s2, s3,
             b1, b2, b3,
             cls.leftmost(arcs, s1), cls.rightmost(arcs, s1),
             cls.leftmost(arcs, s1, 1), cls.rightmost(arcs, s1, 1),
             cls.leftmost(arcs, s2), cls.rightmost(arcs, s2),
             cls.leftmost(arcs, s2, 1), cls.rightmost(arcs, s2, 1),
             cls.leftmost(arcs, cls.leftmost(arcs, s1)),
             cls.leftmost(arcs, cls.leftmost(arcs, s2)),
             cls.rightmost(arcs, cls.rightmost(arcs, s1)),
             cls.rightmost(arcs, cls.rightmost(arcs, s2)),
             ]

    W = [ word.get('form') if word else None for word in words ]
    T = [ word.get('upostag') if word else 'NULL' for word in words ]
    L = [ cls.node_label(arcs, word) if word else 'NULL' for word in words[6:] ]

    return OrderedDict([
                      ('W', W),
                      ('T', T),
                      ('L', L),
                     ])

  @staticmethod
  def node(set, node):
    try:
      set[node]
    except:
      return None
    else:
      return set[node]

  @staticmethod
  def node_label(arcs, elem):
    for a in arcs:
      if a[1]['id'] == elem['id']:
        return a[2]
    return 'NULL'

  @staticmethod
  def leftmost(arcs, elem, index=0):
    if not elem:
      return None

    children = []
    for a in arcs:
      if a[0]['id'] == elem['id']:
        children.append(a[1])

    try:
      return sorted(children, key=lambda node: node['id'])[index]
    except:
      return None

  @staticmethod
  def rightmost(arcs, elem, index=0):
    if not elem:
      return None

    children = []
    for a in arcs:
      if a[0]['id'] == elem['id']:
        children.append(a[1])

    try:
      return sorted(children, key=lambda node: node['id'], reverse=True)[index]
    except:
      return None

  def __str__(self):
    return '\nSTACK => {s}\n\nBUFFER => {b}\n\nARCS => {a}\n'.format(s=self.config['stack'], b=self.config['buffer'], a=self.config['arcs'])
