import os

PATH = os.path.abspath(os.path.dirname(__file__))

PATH_DATA_WORD2VEC = os.path.join(PATH, '../','data/word2vec/')
PATH_CONLLU_TEST = os.path.join(PATH, '../','data/conllu/en-ud-test.conllu')
PATH_CONLLU_TRAIN = os.path.join(PATH, '../','data/conllu/en-ud-train.conllu')
PATH_PARSED_DATA_TEST = os.path.join(PATH, '../','data/trans_systems/test/conllx-{}.npy')
PATH_PARSED_DATA_TRAIN = os.path.join(PATH, '../','data/trans_systems/train/conllx-{}.npy')
PATH_PREDICT_SENTENCE = os.path.join(PATH, '../','data/examples/predict.conllx')
PATH_PREDICT_OUTPUT = os.path.join(PATH, '../','data/examples/predict_out.conllx')
PATH_SAVED_NN = os.path.join(PATH, 'checkpoints/nndep_mlp.params')
PATH_PENN2CONLL = os.path.join(PATH, '../', 'modules/penn-conll-coverter/pennconverter.jar')

VEC_SIZE = 100

NN_EPOCHS = 15
NN_BATCH_SIZE = 25
NN_LEARNING_RATE = 0.01
NN_HIDDEN_UNITS = 200
NN_DROP_OUT = 0.5

DEFAULT_DATASET_SIZE = 40

DEFAULT_CTX = 'cpu'
DEFAULT_VALID = 'uas'

LABELS = {'conllu': {'arc_labels': ['nsubj', 'list', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'det', 'clf', 'case', 'conj', 'cc', 'fixed', 'flat', 'compound', 'parataxis', 'orphan', 'goeswith', 'reparandum', 'punct', 'root', 'dep'],
                     'pos_labels': ['ADJ', 'ADV', 'VERB', 'NOUN', 'PROPN', 'INTJ', 'NUM', 'PRON', 'AUX', 'ADP', 'CCONJ', 'SCONJ', 'DET', 'PART', 'PUNCT', 'SYM', 'X'],
                     'root_label': 'root',
                     'punct_label': 'PUNCT',
                     },
          'conllx': {'arc_labels': ['SUB', 'P', 'APPO', 'AMOD', 'IM', 'NMOD', 'PRN', 'NAME', 'VMOD', 'PMOD', 'SUFFIX', 'DEP', 'CONJ', 'ROOT', 'PRT', 'COORD', 'VC'],
                     'pos_labels': ['PUNCT'] + ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'],
                     'root_label': 'ROOT',
                     'punct_label': 'PUNCT',
                     }
          }

DEFAULT_DP_FORMAT = 'conllx'

ARC_LABELS = LABELS[DEFAULT_DP_FORMAT]['arc_labels']
POS_LABELS = LABELS[DEFAULT_DP_FORMAT]['pos_labels']
ROOT_LABEL = LABELS[DEFAULT_DP_FORMAT]['root_label']