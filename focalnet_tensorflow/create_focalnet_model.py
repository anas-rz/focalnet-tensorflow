from tensorflow import keras
from .focalnet import FocalNet

def Model(img_size=224, **kw) -> keras.Model:

    focalnet_model = FocalNet(img_size=img_size,**kw)

    inputs = keras.Input((img_size, img_size, 3))
    outputs = focalnet_model(inputs)
    final_model = keras.Model(inputs, outputs )

    return final_model



def focalnet_tiny_srf(**kwargs):
    model = Model(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return model

def focalnet_small_srf( **kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return model

def focalnet_base_srf(**kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return model

def focalnet_tiny_lrf(**kwargs):
    model = Model(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_small_lrf(**kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    
    return model

def focalnet_base_lrf(**kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=128, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_tiny_iso_16(**kwargs):
    model = Model(depths=[12], patch_size=16, embed_dim=192, focal_levels=[3], focal_windows=[3], **kwargs)
    return model

def focalnet_small_iso_16(**kwargs):
    model = Model(depths=[12], patch_size=16, embed_dim=384, focal_levels=[3], focal_windows=[3], **kwargs)
    return model

def focalnet_base_iso_16(**kwargs):
    model = Model(depths=[12], patch_size=16, embed_dim=768, focal_levels=[3], focal_windows=[3], use_layerscale=True, use_postln=True, **kwargs)
    return model

# FocalNet large+ models 
def focalnet_large_fl3(**kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_large_fl4(**kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=192, focal_levels=[4, 4, 4, 4], **kwargs)
    return model

def focalnet_xlarge_fl3( **kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[3, 3, 3, 3], **kwargs)
    return model


def focalnet_xlarge_fl4( **kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=256, focal_levels=[4, 4, 4, 4], **kwargs)
    return model

def focalnet_huge_fl3( **kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[3, 3, 3, 3], **kwargs)
    return model

def focalnet_huge_fl4( **kwargs):
    model = Model(depths=[2, 2, 18, 2], embed_dim=352, focal_levels=[4, 4, 4, 4], **kwargs)
    return model
