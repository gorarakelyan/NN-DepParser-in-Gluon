import os

PATH = os.path.abspath(os.path.dirname(__file__))

PATH_DATA_WORD2VEC = os.path.join(PATH, '../','data/word2vec/')
PATH_CONLLU_TEST = os.path.join(PATH, '../','data/conllu/en-ud-test.conllu')
PATH_CONLLU_TRAIN = os.path.join(PATH, '../','data/conllu/en-ud-train.conllu')
PATH_PARSED_DATA_TEST = os.path.join(PATH, '../','data/trans_systems/parsed_test.npy')
PATH_PARSED_DATA_TRAIN = os.path.join(PATH, '../','data/trans_systems/parsed_train.npy')
PATH_PREDICT_SENTENCE = os.path.join(PATH, '../','data/examples/predict.conllu')
PATH_PREDICT_OUTPUT = os.path.join(PATH, '../','data/examples/predict_out.conllu')
PATH_SAVED_NN = os.path.join(PATH, 'checkpoints/nndep_mlp.params')

VEC_SIZE = 100

NN_EPOCHS = 5
NN_BATCH_SIZE = 25
NN_LEARNING_RATE = 0.01
NN_HIDDEN_UNITS = 128
NN_DROP_OUT = 0.5

DEFAULT_CTX = 'cpu'
DEFAULT_VALID = 'uas'

ARC_LABELS = ['nsubj', 'list', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'det', 'clf', 'case', 'conj', 'cc', 'fixed', 'flat', 'compound', 'parataxis', 'orphan', 'goeswith', 'reparandum', 'punct', 'root', 'dep']
POS_LABELS = ['ADJ', 'ADV', 'VERB', 'NOUN', 'PROPN', 'INTJ', 'NUM', 'PRON', 'AUX', 'ADP', 'CCONJ', 'SCONJ', 'DET', 'PART', 'PUNCT', 'SYM', 'X']