import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAvgPool2D, Multiply, Subtract, Concatenate, Dropout, LayerNormalization, Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error

from utils.funcs import ignore_layers
from utils.layers import RBF, CrossAttention
from utils.tf2_keras_vggface.utils import preprocess_input
from utils.tf2_keras_vggface.vggface import VGGFace

# Globals
INPUT_SHAPE = (224, 224,)


class BackboneNetwork():
    
    '''VGG-Face backbone: One of senet50, resnet50, vgg16.'''
    
    def __init__(self, model_name, fine_tune=True):
        # One of senet50, resnet50, vgg16
        self.model_name = model_name
        # Only finetunes the top-most convolutional layer if true
        self.fine_tune = fine_tune
        
    def build_backbone(self):
        
        # Initialize backbone and set trainable layers
        backbone = VGGFace(model=self.model_name, include_top=False)
        for x in backbone.layers:
            x.trainable = False
        if self.fine_tune:
            # Find layers to ignore from backbone architecture.
            ignore_bottom, ignore_top = ignore_layers(self.model_name)
            # Set appropriate layers as trainable.
            for x in backbone.layers[:ignore_bottom]:
                x.trainable = False
            if ignore_top == 0:
                for x in backbone.layers[ignore_bottom:]:
                    x.trainable = True
            else:
                for x in backbone.layers[ignore_bottom:-ignore_top]:
                    x.trainable = True
        
        # Print trainable layers
        for x in backbone.layers:
            print(x.name, x.trainable)
        
        return backbone


class SiameseNetwork(BackboneNetwork):
    
    '''Network encodes images using the same backbone, computes different measures of distance, 
    passes through a bottleneck layers, and yields p(True).'''
    
    def __init__(self, model_name, fine_tune=True, dist='default',
                 loss='binary_crossentropy', metrics=['acc'], optimizer=Adam(0.00002),
                 regularizer=L2(.01), batch_size=16):
        
        self.model_name = model_name
        # Fine tune top-most Conv2D layer of backbone if True.
        self.fine_tune = fine_tune
        # One of 'default', 'rbf', 'cross_attention'
        self.dist = dist
        
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.batch_size = batch_size
        
        super(SiameseNetwork, self).__init__(self.model_name, self.fine_tune)
    
    def build(self):
        
        '''Builds Siamese Network model'''
        
        input_1 = Input(shape=INPUT_SHAPE + (3,))
        input_2 = Input(shape=INPUT_SHAPE + (3,))
        backbone = self.build_backbone()
        
        # Encode images
        x1 = backbone(input_1)
        x2 = backbone(input_2)
        
        # Calculate distance between encodings
        if self.dist == 'rbf':
            x = self.rbf(x1, x2)
        elif self.dist == 'cross_attention':
            x = self.cross_attention(x1, x2)
        else:
            x = self.default(x1, x2)
        
        # Pass through bottleneck.
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.05)(x)    
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.05)(x)
        x = Dense(32, activation="tanh")(x)
        x = Dropout(0.05)(x)    
        out = Dense(1, kernel_regularizer=self.regularizer, activation="sigmoid")(x)
        
        # Define model
        model = Model([input_1, input_2], out)
        # Compile model
        model.compile(loss=self.loss, metrics=self.metrics, optimizer=self.optimizer)
        model.summary()

        return model
    
    def default(self, x1, x2):
        
        '''Default distance encoding'''
        
        x1 = GlobalAvgPool2D()(x1)
        x2 = GlobalAvgPool2D()(x2)

        x1 = LayerNormalization(axis=-1, epsilon=0.001, center=False, scale=False)(x1)
        x2 = LayerNormalization(axis=-1, epsilon=0.001, center=False, scale=False)(x2)
        
        #(exp(-k.||x-y||^2))
        d1 = RBF()(x1, x2)
        # (x-y)^2
        d2 = Subtract()([x1, x2])
        d2 = Multiply()([d2, d2])
        # (x^2-y^2)
        d3_1 = Multiply()([x1, x1])
        d3_2 = Multiply()([x2, x2])
        d3 = Subtract()([d3_1, d3_2])
        # (x.y)
        d4 = Multiply()([x1, x2])
        
        x = Concatenate(axis=-1)([d1, d2, d3, d4])

        return x
    
    def rbf(self, x1, x2):

        '''RBF distance encoding'''
        
        x1 = GlobalAvgPool2D()(x1)
        x2 = GlobalAvgPool2D()(x2)        
        x = RBF()(x1, x2)

        return x

    def cross_attention(self, x1, x2):
        
        '''Cross-Attention distance encoding'''

        cross_attn = CrossAttention(64, (x1.shape[1], x2.shape[2]), x1.shape[1]*x1.shape[2], 
                                    output_dim=64, batch_size=self.batch_size*2)
        x1_ = cross_attn(x1, x2)
        x2_ = cross_attn(x2, x1)
        x1 = Activation('relu')(x1_)
        x2 = Activation('relu')(x2_)
        # Pooling by dot product
        x = tf.einsum('ijklm, ijklm -> ilm', x1, x2)
        x = Flatten()(x)        
        x = LayerNormalization(axis=-1, epsilon=0.001, center=False, scale=False)(x)
        
        return x