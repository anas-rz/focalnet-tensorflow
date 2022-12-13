from tensorflow import keras
import tensorflow.keras.backend as K
from .layers import *

def Mlp(hidden_features=None, dropout_rate=0., act_layer=keras.activations.gelu, out_features=None):
   

    def _apply(x):
        in_features = K.int_shape(x)[-1]
        nonlocal hidden_features, out_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        x = keras.layers.Dense(hidden_features, activation=act_layer)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(out_features, activation=act_layer)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        return x

    return _apply

def PatchEmbed(img_size=(224, 224), patch_size=4, embed_dim=96, use_conv_embed=False, norm_layer=None, is_stem=False):
    

    def _apply(x, H, W):
        nonlocal patch_size
        patch_size = (patch_size, patch_size)
        if use_conv_embed:
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            
            x = keras.layers.ZeroPadding2D(padding=padding)(x)
            x = keras.layers.Conv2D(embed_dim, kernel_size=kernel_size, strides=stride, padding=padding)(x)
        else:
            x = keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(x)
        Ho, Wo, Co = K.int_shape(x)[1], K.int_shape(x)[2], K.int_shape(x)[3]
        x = keras.layers.Reshape((-1, Co))(x)
        if norm_layer is not None:
            x = norm_layer()(x)
        return x, Ho, Wo
    return _apply

def FocalNetBlock(dim, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=keras.activations.gelu, norm_layer=keras.layers.LayerNormalization,
                    focal_level=1, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False, use_postln_in_modulation=False, 
                    normalize_modulator=False, **kwargs):

    def _apply(x, H, W):
        
        C = K.int_shape(x)[-1]
        shortcut = x
        x = x if use_postln else norm_layer()(x)
        x = keras.layers.Reshape((H, W, C))(x)
        x = FocalModulation(dim, proj_drop=drop, focal_window=focal_window, focal_level=focal_level, 
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator)(x)
        x = keras.layers.Reshape((H * W, C))(x)
        x = x if not use_postln else norm_layer()(x)
        x = LayerScale(1e-6, dim)(x)
        x = StochasticDepth(drop_path)(x)
        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.Reshape((H, W, C))(x)
        x = Mlp(hidden_features=dim * mlp_ratio, dropout_rate=drop)(x)
        x = norm_layer()(x)
        if use_postln:
            x_alt = LayerScale(layerscale_value, dim)(x)
            x_alt = StochasticDepth(drop_path)(x_alt)
            x = keras.layers.Add()([x, x_alt])
        x = keras.layers.Reshape((H * W, C))(x)
        return x
    return _apply

def BasicLayer(dim, depth, out_dim, input_resolution,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=keras.layers.LayerNormalization, 
                 downsample=None,  #use_checkpoint=False, 
                 focal_level=1, focal_window=1, 
                 use_conv_embed=False, 
                 use_layerscale=False, layerscale_value=1e-4, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 normalize_modulator=False):
    def _apply(x, H, W):
        for i in range(depth):
            x = FocalNetBlock(
                    dim=dim, 
                    mlp_ratio=mlp_ratio, 
                    drop=drop, 
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    focal_level=focal_level,
                    focal_window=focal_window, 
                    use_layerscale=use_layerscale, 
                    layerscale_value=layerscale_value,
                    use_postln=use_postln, 
                    use_postln_in_modulation=use_postln_in_modulation, 
                    normalize_modulator=normalize_modulator, 
                )(x, H, W)
            # print(x.shape)
        if downsample is not None:
            C = K.int_shape(x)[-1]
            x = keras.layers.Reshape((H, W, C))(x)
            x, Ho, Wo = downsample(img_size=input_resolution, 
                patch_size=2, 
                # in_chans=dim, 
                embed_dim=out_dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False)(x, H, W)
            H, W = Ho, Wo
            
        return x, H, W
    return _apply 
