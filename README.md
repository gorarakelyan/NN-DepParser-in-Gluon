# Transition-based Dependency Parser using Neural Networks
#### Implemented in MXNet/Gluon

### Prerequisite
 - Download word2vec training data and put into folder ```data/word2vec/```
 - Run
```
python utils/word_vec/train.py --train-data={TRAINING_FILE_PATH}
```

### Prepare Data
 - Download English UD Treebank from https://github.com/UniversalDependencies/UD_English/tree/master and put into folder ```data/conllu/```
 - Prepare data running
```
python parser/data.py --file=train,test --max_examples={SENTENCES_TO_PARSE}
```
 - Or you can preprocess seperate files by running
```
python parser/data.py --input_file={INPUT} --output_file={OUTPUT} --max_examples={SENTENCES_TO_PARSE}
```
Set ```SENTENCES_TO_PARSE=-1``` if you don't have any limitations

### Instructions
 - Train parser
```
python parser/train.py --train_data --epochs --batch_size --learning_rate --hidden_units --drop_out --ctx
```
 - Test parser
```
python parser/test.py --test_data --batch_size --hidden_units --drop_out --ctx
```

### Parsing Single Sentence
```
python parser/parse_sentence.py --input_file --hidden_units --drop_out --ctx
```

Also you can configure all the parameters in ```config.py``` file.
