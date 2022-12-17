import torch
from focalnet_tensorflow import *
import tensorflow as tf
tf.keras.backend.set_floatx('float16')

checkpoint = torch.load('weights/focalnet_large_lrf_384_fl4.pth', map_location=torch.device('cpu'))
model = focalnet_large_fl4() # so far srf is making error

keys = checkpoint['model'].keys()
tf_layers = []
keys_first = [key.split('.')[0] for key in keys]
get_first = lambda x: x.split('.')[0]
# keys_first = dict.fromkeys(keys_first)
# patch_embed
patch_embed = [key for key in keys if get_first(key) == 'patch_embed']
part_names = [('.').join(part.split('.')[:-1]) for part in patch_embed]
part_names = set(part_names)
# print(part_names)
for part in part_names:
    layer = model.get_layer(part)
    
    if isinstance(layer, tf.keras.layers.Conv2D):
        weights = checkpoint['model'][f"{part}.weight"].numpy().transpose(2, 3, 1, 0)
        bias = checkpoint['model'][f"{part}.bias"].numpy()
        layer.kernel.assign(
            tf.Variable(weights)
        )
        layer.bias.assign(tf.Variable(bias))
        # print("Assigned Weights and biases")
    elif isinstance(layer, tf.keras.layers.LayerNormalization):
        weights = checkpoint['model'][f"{part}.weight"].numpy()
        bias = checkpoint['model'][f"{part}.bias"].numpy()
        layer.gamma.assign(tf.Variable(weights))
        layer.beta.assign(tf.Variable(bias))
        # print("Assigned norm too!")
    # print(part)
        
    # print(model.get_layer(part_name))

layers = [key for key in keys if get_first(key) == 'layers']
layers_blocks = [key for key in layers if key.split('.')[2] == 'blocks']
fourth = [key.split('.')[4] for key in layers_blocks]
fourth = set(fourth)
# print(fourth)

# modulation
layers_blocks_modulation = [key for key in layers_blocks if key.split('.')[4] == 'modulation']
# print(layers_blocks_modulation)
layers_blocks_modulation = [ ('.').join(key.split('.')[:5]) for key in layers_blocks_modulation]
# 'layers.2.blocks.3.modulation.focal_layers.1.0.weight', 'layers.2.blocks.4.modulation.f.weight', 'layers.2.blocks.4.modulation.f.bias', 'layers.2.blocks.4.modulation.h.weight', 'layers.2.blocks.4.modulation.h.bias', 'layers.2.blocks.4.modulation.proj.weight', 'layers.2.blocks.4.modulation.proj.bias', 'layers.2.blocks.4.modulation.focal_layers.0.0.weight', 'layers.2.blocks.4.modulation.focal_layers.1.0.weight', 'layers.2.blocks.5.modulation.f.weight'
# focal_layers, f, h, proj, 
for part in layers_blocks_modulation:
    layer = model.get_layer(part)
    # f
    weights_f = checkpoint['model'][f"{part}.f.weight"].numpy().transpose(1, 0)
    # print(weights_f.shape)
    # print(layer.f.kernel.shape)
    bias_f = checkpoint['model'][f"{part}.f.bias"].numpy()
    layer.f.kernel.assign(
        tf.Variable(weights_f)
    )
    layer.f.bias.assign(tf.Variable(bias_f))
    # print("done for f")
    # h
    weights_h = checkpoint['model'][f"{part}.h.weight"].numpy().transpose(2, 3, 1, 0)
    bias_h = checkpoint['model'][f"{part}.h.bias"].numpy()
    layer.h.kernel.assign(
        tf.Variable(weights_h)
    )
    layer.h.bias.assign(tf.Variable(bias_h))
    # print("Assigned for h")
    # proj
    weights_proj = checkpoint['model'][f"{part}.proj.weight"].numpy().transpose(1, 0)
    bias_proj = checkpoint['model'][f"{part}.proj.bias"].numpy()
    layer.proj.kernel.assign(
        tf.Variable(weights_proj)
    )
    layer.proj.bias.assign(tf.Variable(bias_proj))
    # print("done for proj")

# print(layers_blocks)
layers_blocks_norm1 = [key for key in layers_blocks if key.split('.')[4] == 'norm1']
layers_blocks_norm1 = [ ('.').join(key.split('.')[:-1]) for key in layers_blocks_norm1]
for part in layers_blocks_norm1:
    layer = model.get_layer(part)
    weights = checkpoint['model'][f"{part}.weight"].numpy()
    bias = checkpoint['model'][f"{part}.bias"].numpy()
    layer.gamma.assign(tf.Variable(weights))
    layer.beta.assign(tf.Variable(bias))
    # print("Assigned Weights")

layers_blocks_norm2 = [key for key in layers_blocks if key.split('.')[4] == 'norm2']
layers_blocks_norm2 = [ ('.').join(key.split('.')[:-1]) for key in layers_blocks_norm2]
for part in layers_blocks_norm2:
    layer = model.get_layer(part)
    weights = checkpoint['model'][f"{part}.weight"].numpy()
    bias = checkpoint['model'][f"{part}.bias"].numpy()
    layer.gamma.assign(tf.Variable(weights))
    layer.beta.assign(tf.Variable(bias))
    # print("Assigned Weights")

layers_blocks_mlp = [key for key in layers_blocks if key.split('.')[4] == 'mlp']

# print(layers_blocks_norm1)
# layers.3.blocks.1.norm2
layers_blocks_mlp = [('.').join(key.split('.')[:-1]) for key in layers_blocks_mlp]
layers_blocks_mlp = set(layers_blocks_mlp)


# print(layers_blocks_mlp)
# print(model.get_layer('layers.0.blocks.0.mlp.fc1'))
for part in layers_blocks_mlp:
    layer = model.get_layer(part)
    weights = checkpoint['model'][f"{part}.weight"].numpy().transpose(1, 0)
    bias = checkpoint['model'][f"{part}.bias"].numpy()
    layer.kernel.assign(
        tf.Variable(weights)
    )
    layer.bias.assign(tf.Variable(bias))
    # print("Added for MLPs")




# downsampling layers
layers_downsample = [key for key in layers if key.split('.')[2] == 'downsample']
layers_downsample = [('.').join(key.split('.')[:-1]) for key in layers_downsample ]
layers_downsample = set(layers_downsample)
# print(layers_downsample)
# print(layers_downsample)
for part in layers_downsample:
    layer = model.get_layer(part)
    if isinstance(layer, tf.keras.layers.Conv2D):
        weights = checkpoint['model'][f"{part}.weight"].numpy().transpose(2, 3, 1, 0)
        bias = checkpoint['model'][f"{part}.bias"].numpy()
        layer.kernel.assign(
            tf.Variable(weights)
        )
        layer.bias.assign(tf.Variable(bias))
        # print("Assigned Weights and biases")
    elif isinstance(layer, tf.keras.layers.LayerNormalization):
        weights = checkpoint['model'][f"{part}.weight"].numpy()
        bias = checkpoint['model'][f"{part}.bias"].numpy()
        layer.gamma.assign(tf.Variable(weights))
        layer.beta.assign(tf.Variable(bias))
        # print("Assigned Weights and biases")
    # print(model.get_layer(layer))
# layers = 
# Print unique at third level


# unique_layers = []
# for l in layers:
#     unique_layers.append(".".join(l.split('.')[:2]))
# unique_layers = set(unique_layers)
# print(unique_layers)

norm_layer = [key for key in keys if get_first(key) == 'norm']
norm_layer = [key.split('.')[0] for key in norm_layer]
norm_layer = set(norm_layer)
for n in norm_layer:
    layer = model.get_layer(n)
    weights = checkpoint['model'][f"{n}.weight"].numpy()
    bias = checkpoint['model'][f"{n}.bias"].numpy()
    layer.gamma.assign(tf.Variable(weights))
    layer.beta.assign(tf.Variable(bias))
    # print("Added Normalization Weights")

heads = [key for key in keys if get_first(key) == 'head']
heads = [key.split('.')[0] for key in heads]
heads = set(heads)
for part in heads:
    layer = model.get_layer(part)
    weights = checkpoint['model'][f"{part}.weight"].numpy().transpose(1, 0)
    bias = checkpoint['model'][f"{part}.bias"].numpy()
    layer.kernel.assign(
        tf.Variable(weights)
    )
    layer.bias.assign(tf.Variable(bias))
 
# print(patch_embed)
# # print(layers)
# print(norms)
# print(heads)
# # 
