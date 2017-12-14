import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon as g

from config import *

class MLP:
  def __init__(self, *args, **kwargs):
    self.hidden_units = kwargs.get('hidden_units')
    self.drop_out = kwargs.get('drop_out')
    self.mlp = None

    self.model()

  def model(self):
    net = g.nn.Sequential()
    with net.name_scope():
      net.add(g.nn.Dense(self.hidden_units, activation='relu'))
      net.add(g.nn.Dropout(self.drop_out))
      net.add(g.nn.Dense(len(ARC_LABELS)*2+1))

    self.mlp = net

  def train(self, train_data, epochs, learning_rate):
    net = self.mlp

    net.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)

    criterion = g.loss.SoftmaxCrossEntropyLoss()
    trainer = g.Trainer(net.collect_params(), 'sgd',
                          {'learning_rate': learning_rate})
    
    print('Log: Training started..')

    for e in range(epochs):
      cumulative_loss = 0
      for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(self.ctx)
        label = label.as_in_context(self.ctx)
        with autograd.record():
          output = net(data)
          loss = criterion(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()

      train_accuracy = self.evaluation(train_data)
      print('Log: Epoch {e}. Train Accuracy - {acc}'.format(e=e, acc=train_accuracy))

    print('Log: Training done!')

    print('Log: Savind model..')

    net.save_params(PATH_SAVED_NN)

    print('Log: Saved!')

  def predict(self, data):
    return self.mlp(data)

  def load_model(self):
    self.mlp.load_params(PATH_SAVED_NN, ctx=self.ctx)

  def evaluation(self, data_iterator):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
      data = data.as_in_context(self.ctx)
      label = label.as_in_context(self.ctx)
      output = self.mlp(data)
      predictions = nd.argmax(output, axis=1)
      acc.update(preds=predictions, labels=label)
    return acc.get()[1]

  def set_ctx(self, context):
    context = context
    if context == 'cpu':
      self.ctx = mx.cpu()
    else:
      self.ctx = mx.gpu()

  @staticmethod
  def prepare_data(path, batch_size, train):
    print('Log: Preparing data..')

    input = path

    data = np.load(input)

    dataset = []
    for row in data:
      for i in row:
        d = np.concatenate(tuple((np.array(c).flatten() for c in i[0])))
        dataset.append((d.astype(np.float32), i[1]))

    return g.data.DataLoader(dataset, batch_size, shuffle=True if train else False)
