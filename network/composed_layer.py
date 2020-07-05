from tensorflow.keras import layers
from data.preprocess import Preprocessor
from javalang.tree import MethodDeclaration
import numpy as np


class ComposedLayer(layers.Layer):
    def __init__(self, emb_input_dim, emb_output_dim, lstm_num, **kwargs):
        super(ComposedLayer, self).__init__(**kwargs)
        self._embedding = layers.Embedding(emb_input_dim, emb_output_dim,mask_zero=True)
        self._lstm = layers.LSTM(lstm_num)

    def call(self, inputs, **kwargs):
        embedded = self._embedding(inputs)
        mask = self._embedding.compute_mask(inputs)
        output = self._lstm(embedded, mask=mask)
        return output


max_seq_length = 500
np.random.seed(7)

preprocessor = Preprocessor(
    root="C:\\Users\\Yan\\IdeaProjects\\psn-adapter-service",
    node_type=MethodDeclaration,
    max_seq_length=max_seq_length
)
parsed = preprocessor.parse_all()
layers = ComposedLayer(preprocessor.get_dict_size() + 1, 16, 32)
print(layers(parsed))
