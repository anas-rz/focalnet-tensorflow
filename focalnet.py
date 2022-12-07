import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

class FocalModulation(keras.layers.Layer):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=True, normalize_modulator=False):
        super(FocalModulation, self).__init__()
        self.focal_level = focal_level
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.f = keras.layers.Dense(2*dim + (focal_level+1), use_bias=bias)
        self.h = keras.layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias)

        self.act = keras.activations.gelu
        self.proj = keras.layers.Dense(dim)
        self.proj_drop = keras.layers.Dropout(proj_drop)
        self.focal_layers = []
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = focal_factor*k + focal_window
            self.focal_layers.append(
                keras.Sequential(
                    [keras.layers.Conv2D(dim, kernel_size=kernel_size, strides=1, groups=dim, use_bias=False, padding="Same"),
                    keras.layers.Lambda(lambda x: keras.activations.gelu(x)),
                    ]
                ) )             
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = keras.layers.LayerNormalization()

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

class Mlp(keras.layers.Layer):
    def __init__(self, act_layer, hidden_features=None, out_features=None, dropout_rate=0., **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.act_layer = act_layer
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.hidden_features = self.hidden_features or self.in_features
        self.out_features = self.out_features or self.in_features
        self.model =  keras.Sequential([keras.layers.Dense(self.hidden_features), self.act_layer, keras.layers.Dropout(self.dropout_rate), keras.layers.Dense(self.in_features), 
                                        self.act_layer, keras.layers.Dropout(self.dropout_rate)])
    def call(self, x):
        return self.model(x)


class IdentityLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs)


    def call(self, x):
        return x


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


class FocalNetBlock(keras.layers.Layer):
    r""" Focal Modulation Network Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=keras.activations.gelu, norm_layer=keras.layers.LayerNormalization,
                    focal_level=1, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False, use_postln_in_modulation=False, 
                    normalize_modulator=False):
        super().__init__()
        self.norm1 = norm_layer()
        self.modulation = FocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=focal_level, 
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator
        )
        self.use_postln = use_postln
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else IdentityLayer()
        self.norm2 = norm_layer()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.act_layer = act_layer
        self.drop = drop
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = tf.Variable(layerscale_value * tf.ones((dim)), trainable=True)
            self.gamma_2 = tf.Variable(layerscale_value * tf.ones((dim)), trainable=True)


        
    def build(self, input_shape):
        self.mlp = Mlp(keras.layers.Lambda(self.act_layer), hidden_features=self.mlp_hidden_dim, dropout_rate=self.drop)

        
        

    def call(self, x):
        B, L, C = x.shape
        shortcut = x
        x = x if self.use_postln else self.norm1(x)
        x = tf.reshape(x, (B, self.H, self.W, C))
        x = self.modulation(x)
        x = tf.reshape(x, (B, self.H * self.W, C))
        x = x if not self.use_postln else self.norm1(x)
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = tf.reshape(x, (B, self.H, self.W, C))
 
        x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))
        x = tf.reshape(x, (B, self.H * self.W, C))
        return x


class BasicLayer(tf.keras.Model):
    """ A basic Focal Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, depth, out_dim, input_resolution,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=keras.layers.LayerNormalization, 
                 downsample=None,  #use_checkpoint=False, 
                 focal_level=1, focal_window=1, 
                 use_conv_embed=False, 
                 use_layerscale=False, layerscale_value=1e-4, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 normalize_modulator=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_postln = use_postln
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator


        self.blocks = [
            FocalNetBlock(
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
            )
            for i in range(self.depth)]
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, 
                patch_size=2, 
                in_chans=dim, 
                embed_dim=out_dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False
            )
        else:
            self.downsample = None
    def call(self, x, H, W):
        # print(x.shape)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)

        if self.downsample is not None:
            x = tf.transpose(x, (0, 1, 2))
            x = tf.reshape(x, (x.shape[0], H, W, -1))
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W        
        return x, Ho, Wo

class PatchEmbed(keras.layers.Layer):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=None, is_stem=False):
        super().__init__()
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim

        if use_conv_embed:
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            
            self.proj = keras.Sequential([keras.layers.ZeroPadding2D(padding=padding),
                keras.layers.Conv2D(embed_dim, kernel_size=kernel_size, strides=stride, padding=padding)])
        else:
            self.proj = keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer()
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.shape
        x = self.proj(x)        
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, (B, H * W, C))
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W


class FocalNet(tf.keras.Model):
    r""" Focal Modulation Networks (FocalNets)
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=keras.layers.LayerNormalization, 
                patch_norm=True,
                use_checkpoint=False,                 
                focal_levels=[2, 2, 2, 2], 
                focal_windows=[3, 3, 3, 3], 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_postln=False, 
                use_postln_in_modulation=False, 
                normalize_modulator=False, 
                **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=(img_size, img_size), 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if self.patch_norm else None, 
            is_stem=True)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = keras.layers.Dropout(drop_rate)

        # stochastic depth
        dpr = [x.numpy() for x in tf.linspace(0., drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # build layers
        self.layers_custom = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer], 
                            out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               use_conv_embed=use_conv_embed,
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value, 
                               use_postln=use_postln,
                               use_postln_in_modulation=use_postln_in_modulation, 
                               normalize_modulator=normalize_modulator
                    )
            self.layers_custom.append(layer)
        self.norm = norm_layer()
        self.avgpool = tfa.layers.AdaptiveAveragePooling1D(1)
        self.head = keras.layers.Dense(num_classes) if num_classes > 0 else None
        self.flatten = keras.layers.Flatten()

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers_custom:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x)  # B C 1
        x = self.flatten(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def focalnet_tiny_srf(**kwargs):
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return model

def focalnet_small_srf( **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return model

def focalnet_base_srf(**kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return model

def focalnet_tiny_lrf(**kwargs):
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_small_lrf(**kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    
    return model

def focalnet_base_lrf(**kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=128, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_tiny_iso_16(**kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=192, focal_levels=[3], focal_windows=[3], **kwargs)
    return model

def focalnet_small_iso_16(**kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=384, focal_levels=[3], focal_windows=[3], **kwargs)
    return model

def focalnet_base_iso_16(**kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=768, focal_levels=[3], focal_windows=[3], use_layerscale=True, use_postln=True, **kwargs)
    return model

# FocalNet large+ models 
def focalnet_large_fl3(**kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_large_fl4(**kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[4, 4, 4, 4], **kwargs)
    return model

def focalnet_xlarge_fl3( **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[3, 3, 3, 3], **kwargs)
    return model


def focalnet_xlarge_fl4( **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[4, 4, 4, 4], **kwargs)
    return model

def focalnet_huge_fl3( **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_huge_fl4( **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[4, 4, 4, 4], **kwargs)
    return model


if __name__ == '__main__':
    print("A tensorflow implementation of FocalNet..")
