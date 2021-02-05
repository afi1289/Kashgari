# the original model is as same as BiGRU, therefore I changed it
# the original code is from:
# https://github.com/CHI-MING-LEE/2020_Esun-Tbrain_NLP_Competition/blob/master/2_Build_BERT_Model.ipynb

# file: cnn_lstm_model.py
# time: 5:28 下午

from typing import Dict, Any

from tensorflow import keras

from kashgari.layers import L
from kashgari.tasks.labeling.abc_model import ABCLabelingModel

        
class LSTM_CNN_Model(ABCLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.25
            },
            'layer_conv1d':{
                'kernel_size': 10,
                'padding': 'same',
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define your layers
        layer_blstm1 = L.Bidirectional(L.LSTM(**config['layer_blstm1']),
                                       name='layer_blstm1')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_conv1d = L.Conv1D(output_dim, **config['layer_conv1d'])
        layer_gmp = L.GlobalMaxPooling1D()
        layer_bn = L.BatchNormalization()
        layer_activation = L.Activation(**config['layer_activation'])

        # Define tensor flow
        tensor = layer_blstm1(embed_model.output)
        tensor = layer_dropout(tensor)
        tensor = layer_conv1d(tensor)
        # tensor = layer_gmp(tensor)
        output_tensor = layer_activation(tensor)
        # 最後維度 100 * 4

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)


if __name__ == "__main__":
    pass
