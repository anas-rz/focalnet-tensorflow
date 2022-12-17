import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from .layers import *
from .blocks import *

def FocalNet(img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=128, 
                depths=[2, 2, 6, 2
                        ], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=keras.layers.LayerNormalization, 
                patch_norm=True,
                use_checkpoint=False,                 
                focal_levels=[2, 2, 3, 2], 
                focal_windows=[3, 2, 3, 2], 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_postln=False, 
                use_postln_in_modulation=False, 
                normalize_modulator=False):
    num_layers = len(depths)
    embed_dim = [embed_dim * (2 ** i) for i in range(num_layers)]
    dpr = [x.numpy() for x in tf.linspace(0., drop_path_rate, sum(depths))]  # stochastic depth decay rule

    
    def _apply(x):
        nonlocal num_classes
        x, *patches_resolution = PatchEmbed(
            img_size=(img_size, img_size), 
            patch_size=patch_size, 
            # in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if patch_norm else None, 
            is_stem=True)(x, img_size, img_size)
        H, W = patches_resolution[0], patches_resolution[1]
        x = keras.layers.Dropout(drop_rate)(x)
        for i_layer in range(num_layers):
            x, H, W = BasicLayer(dim=embed_dim[i_layer], 
                            out_dim=embed_dim[i_layer+1] if (i_layer < num_layers - 1) else None,  
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio,
                               drop=drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               downsample=PatchEmbed if (i_layer < num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               use_conv_embed=use_conv_embed,
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value, 
                               use_postln=use_postln,
                               use_postln_in_modulation=use_postln_in_modulation, 
                               normalize_modulator=normalize_modulator
                    )(x, H, W)
        x = norm_layer(name='norm')(x)  # B L C
        x = tfa.layers.AdaptiveAveragePooling1D(1)(x)
        x = keras.layers.Flatten()(x)
        num_classes = num_classes if num_classes > 0 else None
        x = keras.layers.Dense(num_classes, name='head')(x)
        return x

    return _apply