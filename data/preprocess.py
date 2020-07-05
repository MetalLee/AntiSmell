import javalang.parse as parse
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.dictionary import NodeTypeDictionary


class Preprocessor(object):

    def __init__(self, root,
                 node_type=None,
                 dictionary=None,
                 max_seq_length=500):
        self._root = root
        self._node_type = node_type
        self._dictionary = NodeTypeDictionary(dictionary)
        self._max_seq_length = max_seq_length

    def parse_all(self):
        result = list()
        for root, dirs, files in os.walk(self._root):
            for file in files:
                if os.path.splitext(file)[-1] == '.java':
                    result.extend(self._parse_file(os.path.join(root, file)))
        padded_result = pad_sequences(result, padding='post', maxlen=self._max_seq_length,truncating='post')
        return padded_result

    def _parse_file(self, file):
        full_path = os.path.join(file)
        file_result = list()
        with open(full_path, 'r') as file:
            file_str = file.read()
            cu = parse.parse(file_str)
            for path, node in cu.filter(self._node_type):
                children = self._extract_children(node)
                file_result.append(children)
        return file_result

    def _extract_children(self, root):
        children = list()
        for path, child in root:
            if child is not root:
                type_name = type(child).__name__
                children.append(self._dictionary.look_up(type_name))
        return children

    def get_dict_size(self):
        return self._dictionary.size()