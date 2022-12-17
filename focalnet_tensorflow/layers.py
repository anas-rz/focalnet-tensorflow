import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class FocalModulation(keras.layers.Layer):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=True, normalize_modulator=False, prefix=None):
        
        if prefix is not None:
            prefix = prefix + ".modulation"
            name = prefix #+ str(int(K.get_uid(prefix)) - 1)
        else:
            name = "focal_modulation"
        
        super(FocalModulation, self).__init__(name=name)
        self.focal_level = focal_level
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        
        self.f = keras.layers.Dense(2*dim + (focal_level+1), use_bias=bias, name=f'{name}.f')
        
        self.h = keras.layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias, name=f'{name}.h')
        
        self.act = keras.activations.gelu
        self.proj = keras.layers.Dense(dim, name=f'{name}.proj')
        self.proj_drop = keras.layers.Dropout(proj_drop)
        self.map = {f"{name}.f": self.f, f'{name}.h': self.h, f'{name}.proj': self.proj}

        self.focal_layers = []
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            _name = f'{prefix}.focal_layers.'
            _name = _name + str(K.get_uid(_name) - 1)
            # print(name)
            kernel_size = focal_factor*k + focal_window
            _layer = keras.layers.Conv2D(dim, kernel_size=kernel_size, strides=1, groups=dim, use_bias=False, 
            padding="Same", activation=self.act, name=_name)
            self.map[_name] = _layer
            self.focal_layers.append(_layer)            
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = keras.layers.LayerNormalization(name=f'{prefix}.norm')
            self.map['norm'] = self.ln
        # print(len(self.map.keys()))
    
    def call(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]
        x = self.f(x)
        q, ctx, self.gates = tf.split(x, (C, C, self.focal_level+1), -1)
        ctx_all = 0 
        for l in range(self.focal_level):  
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + tf.math.multiply(ctx, self.gates[:,: , :, l:l+1])
        ctx = tf.math.reduce_mean(ctx, 1, keepdims=True)
        ctx = tf.math.reduce_mean(ctx, 2, keepdims=True)
        ctx_global = self.act(ctx)
        ctx_all = ctx_all + ctx_global*self.gates[:,: , :, self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)
        modulator = self.h(ctx_all)
        x_out = q*modulator
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def _get_layer(self, name):
        return self.map[name]

        
class LayerScale(keras.layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

class StochasticDepth(keras.layers.Layer):
    """
    https://keras.io/examples/vision/cct/
    """
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
