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

  def train(self, data_attr, epochs, learning_rate):
    net = self.mlp

    net.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)

    criterion = g.loss.SoftmaxCrossEntropyLoss()
    trainer = g.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': learning_rate})
    
    print('Log: Training started..')

    best_model = 0

    for e in range(epochs):
      cumulative_accuracy = 0
      set_count = 0
      data_gen = self.prepare_data(**data_attr)
      #cumulative_loss = 0
      for train_data in data_gen:
        print('Log: Training current dataset..')
        set_count += 1
        for i, (data, label) in enumerate(train_data):
          data = data.as_in_context(self.ctx)
          label = label.as_in_context(self.ctx)
          with autograd.record():
            output = net(data)
            loss = criterion(output, label)
          loss.backward()
          trainer.step(data.shape[0])
          #cumulative_loss += nd.sum(loss).asscalar()

        cumulative_accuracy += self.evaluation(train_data)
      train_accuracy = cumulative_accuracy / set_count
      print('Log: Epoch {e}. Train Accuracy - {acc}'.format(e=e, acc=train_accuracy))

      if train_accuracy > best_model:
        net.save_params(PATH_SAVED_NN)
        best_model = train_accuracy

    print('Log: Training done!')

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
  def prepare_data(path, dataset_size, batch_size, shuffle_data):
    index = 0

    while True:
      dataset = []
      print('Log: Preparing dataset {i} ..'.format(i=index))
      while True:
        try:
          data = np.load(path.format(index))
        except:
          break

        index += 1

        for row in data:
          for i in row:
            d = np.concatenate(tuple((np.array(c).flatten() for c in i[0])))
            dataset.append((d.astype(np.float32), i[1]))
      
        if index%dataset_size == 0:
          break

      if dataset:
        yield g.data.DataLoader(dataset, batch_size, shuffle=True if shuffle_data else False)
      else:
        break
  