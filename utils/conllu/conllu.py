import re
from collections import defaultdict, namedtuple, OrderedDict

class CONLLU:
  DEFAULT_FIELDS = ('id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc')
  
  def parse(self, text, fields=DEFAULT_FIELDS):
    parsed = [
      [
        self.parse_line(line, fields)
        for line in sentence.split('\n')
        if line and not line.strip().startswith('#')
      ]
      for sentence in text.split('\n\n') if sentence
    ]
    return parsed

  def parse_tree(self, text):
    result = self.parse(text)
    print(result)

    trees = []
    for sentence in result:
      head_indexed = defaultdict(list)
      for token in sentence:
        head_indexed[token['head']].append(token)

      trees += self.create_tree(head_indexed)
    return trees

  def create_tree(self, node_children_mapping, start=0):
    TreeNode = namedtuple( 'TreeNode', [
      'data', 
      'children'
    ])
    
    subtree = [
      TreeNode(child, self.create_tree(node_children_mapping, child['id']))
      for child in node_children_mapping[start]
    ]
    return subtree

  def parse_line(self, line, fields=DEFAULT_FIELDS):
    line = re.split(r'\t| {2,}', line)
    data = OrderedDict()

    for i, field in enumerate(fields):
      if i >= len(line):
        break

      if field == 'id':
        value = self.parse_int_value(line[i])
      elif field == 'xpostag':
        value = self.parse_nullable_value(line[i])
      elif field == 'feats':
        value = self.parse_dict_value(line[i])
      elif field == 'head':
        value = self.parse_int_value(line[i])
      elif field == 'deprel':
        value = line[i]
        if ':' in value:
          value = value.split(':')[0]
      elif field == 'deps':
        value = self.parse_nullable_value(line[i])
        if value:
          value = value.split(':')[0]
      elif field == 'misc':
        value = self.parse_dict_value(line[i])
      else:
        value = line[i]

      data[field] = value

    return data
  
  def parse_int_value(self, value):
    if value.isdigit():
      return int(value)
    return None
  
  def parse_dict_value(self, value):
    if '=' in value:
      if value and value.split('|'):
        parts = []
        for part in value.split('|'):
          if len(part.split('=')) == 2:
            parts.append((part.split('=')[0], self.parse_nullable_value(part.split('=')[1])))
        return parts

    return self.parse_nullable_value(value)

  def parse_nullable_value(self, value):
    if not value or value == '_':
      return 
    return value