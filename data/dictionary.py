import javalang.tree as tree
import inspect


class NodeTypeDictionary:

    def __init__(self, dictionary=None, start_from=1):
        if dictionary is None or not isinstance(dictionary, dict):
            self._dictionary = dict()
            node_classes = inspect.getmembers(tree, inspect.isclass)
            index = start_from
            for name, node_class in node_classes:
                self._dictionary[name] = index
                index += 1
        else:
            self._dictionary = dictionary.copy()

    def __str__(self):
        return str(self._dictionary)

    def size(self):
        return len(self._dictionary)

    def look_up(self, node_type):
        return self._dictionary.get(node_type)