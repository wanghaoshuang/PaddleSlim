"""
Zigzag Permutation for roatation, according to weight magnitude.
"""
import paddle
from ...utils.log import logger
__all__ = ['permute_according_weight']

def importance_degree(W):
    """
    L1 norm as weight magnitude
    """
    abs_max = paddle.max(paddle.abs(W), axis=1)
    return abs_max.cast('float32').tolist()

def get_permutation_zigzag(W, block_size=512):
    """
    permutation according to weight magnitude
    """
    hidden_dim = W.shape[0]
    # get the magnitude of each weight
    degree = importance_degree(W)
    pairs = [(i, degree[i]) for i in range(hidden_dim)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    # zigzag permutation, for example: there are 3 blocks and each block has 3 elements, we will get:
        # p1: [1,6,7]
        # p2: [2,5,8]
        # p3: [3,4,9]
    pairs = _zigzag(pairs, hidden_dim, block_size)
    for i in range(len(pairs)):
        pairs[i].sort(key=lambda x: x[1], reverse=True)
    
    perm = paddle.zeros(hidden_dim, dtype='int32')
    for i in range(len(pairs)):
        perm[i * block_size:(i+1) * block_size] = paddle.to_tensor([_[0] for _ in pairs[i]], dtype='int32')
        
    perm_w = W[perm, :]
    return perm_w, perm

    
def _zigzag(pairs, hidden_dim, block_size=512):
    """
    zigzag permutation
    """
    cur = 0
    up = True
    l = [[] for i in range(hidden_dim // block_size)]
    for i in range(len(pairs)):
        l[cur].append(pairs[i])
        if up:
            cur += 1
            if cur == len(l):
                cur -= 1
                up = False
        else:
            cur -= 1
            if cur == -1:
                cur += 1
                up = True
    return l

def add_ffn2_R4_hook(linear, rotation_utils):
    """
    Add FFN2 hooks into Linear Layer
    """
    ffn2_hook = rotation_utils['ffn2_hook']
    linear.register_forward_pre_hook(ffn2_hook)
    

def permutation_one_layer(layer, rotation_utils=None):
    """
    Permutation operation on single-layer decoders and perform R4 rotation
    args:
        layer: single TransformerDecoderLayer
        rotation_utils: a dict contains R4 and ffn2_hook,default as None, which means no R4 rotation
    return:
        layer: a permuted and rotated(if needed) TransformerDecoderLayer
    """
    logger.debug(f'[Hadamard] Add permutation(according weight) on {layer.full_name()}')
    
    if rotation_utils is not None:
        R4 = rotation_utils['R4']
    else:
        R4 = None
    R4_size = layer.linear2.weight.shape[0]
    new_linear2_weight = layer.linear2.weight 
    new_linear2_weight,perm = get_permutation_zigzag(new_linear2_weight)
    if R4 is not None:
        new_linear2_weight = R4.cast(new_linear2_weight.dtype).T @ new_linear2_weight
    layer.linear2.weight.set_value(new_linear2_weight)
    
    if hasattr(layer.linear2,'smooth_weight'):
        new_smooth_weight = layer.linear2.smooth_weight[perm]
        layer.linear2.smooth_weight.set_value(new_smooth_weight)
        
    new_linear1_weight = layer.linear1.weight
    linear1_perm = paddle.zeros(R4_size*2, dtype='int32')
    for i in range(len(perm)):
        linear1_perm[2*i] = perm[i] * 2
        linear1_perm[2*i+1] = perm[i] * 2 + 1

    new_linear1_weight = new_linear1_weight[:,linear1_perm]
    new_linear1_weight = new_linear1_weight
    layer.linear1.weight.set_value(new_linear1_weight)
    
    if hasattr(layer.linear1, 'bias') and layer.linear1.bias is not None:
        new_linear1_bias = layer.linear1.bias[linear1_perm]
        layer.linear1.bias.set_value(new_linear1_bias)
    
    if rotation_utils is not None:
        if hasattr(layer.linear2, 'layer'):
            add_ffn2_R4_hook(layer.linear2.layer, rotation_utils)
            logger.debug(f'[Hadamard] Adding hadamard rotation hook on linear2.layer')
        else:
            add_ffn2_R4_hook(layer.linear2, rotation_utils)
            logger.debug(f'[Hadamard] Adding hadamard rotation hook on linear2')
    return layer
    

def permute_according_weight(model, rotation_utils=None, is_layer=False):
    """
    Permutation operation on model or single-layer decoders and perform R4 rotation(if needed)
    args:
        model: model or single TransformerDecoderLayer
        rotation_utils: a dict contains R4 and ffn2_hook, default as None, which means no R4 rotation
        is_layer: whether the input is a single layer. if True, permutation will be performed on the input layer.
        
    return:
        model: a permuted and rotated(if needed) model
    """
    if is_layer:
        return permutation_one_layer(model, rotation_utils)
    else:
        layers = model.gpt.decoder.layers
        for layer in layers:
            permutation_one_layer(layer, rotation_utils)
    
    return model
