

from nn.pkg.blocks.conv_block import ConvBlock
from nn.pkg.blocks.conv_block import filter_size
from nn.pkg.layers.fc.layer import FullyConnected
from nn.pkg.layers.yolo_output.layer import YoloOutput


class TinysimmoYOLOModel:
    def __init__(self, model_path):
        self.model_path = model_path

    def create_new_model(self):
        self.conv_blocks = [
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, input_channels=3),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, input_channels=16),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, input_channels=32),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, input_channels=64),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, input_channels=64),
        ]
        
        S = 4
        B = 2
        C = 1
        self.fc_layer = FullyConnected(num_input_neurons=filter_size*filter_size*128, num_output_neurons=256) # 256 is a bit more than 4*4(2*5+1)
        self.output_layer = YoloOutput(input_size=256, S=S, B=B, C=C)
        
    def forward(self, X):
        for block in self.conv_blocks:
            X = block.forward(X)
        
        X = X.reshape(X.shape[0], -1)
        X = self.fc_layer.feed_forward(X)
        X = self.output_layer.feed_forward(X)
        
        return X