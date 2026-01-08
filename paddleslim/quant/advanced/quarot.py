# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import json
import tqdm
import paddle
import paddle.distributed as dist
from .rms_norm import RMSNorm
from .hadamard_utils import random_hadamard_matrix


def replace_modules(
    root,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_sublayers():
        new_module = None
        if isinstance(module, type_to_replace):
            if (
                replace_layers
            ):  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.named_sublayers())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


def fuse_ln_linear(layernorm, linear_layers):
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        # Calculating new weight and bias
        W_ = linear.weight.detach()
        W_ = W_.cast("float32")
        paddle.assign(
            (W_.transpose((1, 0)) * layernorm.weight.detach().cast("float32"))
            .cast(linear_dtype)
            .transpose((1, 0)),
            output=linear.weight,
        )
        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = paddle.nn.Parameter(
                    paddle.zeros(linear.out_features, dtype="float32")
                )
            paddle.assign(
                (
                    linear.bias.detach().cast("float32")
                    + paddle.matmul(
                        W_.transpose((1, 0)), layernorm.bias.detach().cast("float32")
                    )
                ).cast(linear_dtype),
                output=linear.bias,
            )
        del W_
    paddle.device.cuda.empty_cache()


def bake_mean_into_linear(linear):
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.detach()
    W_ = W_.cast("float32")
    paddle.assign(
        (W_ - W_.mean(axis=-1, keepdim=True)).cast(linear_dtype), output=linear.weight
    )
    del W_
    if linear.bias is not None:
        b_ = linear.bias.detach()
        b_ = b_.cast("float32")
        paddle.assign((b_ - b_.mean()).cast(linear_dtype), output=linear.bias)
        del b_
    paddle.device.cuda.empty_cache()


def fuse_layer_norms(model):
    """
    将模型中的层规范化（Layer Normalization）和线性层（Linear Layers）进行融合。
    该函数会修改传入的模型，并返回一个新的模型，其中所有的层规范化已经被融合到相应的线性层中。

    Args:
        model (paddle.nn.Layer): 包含层规范化和线性层的模型。

    Returns:
        None, str: 不返回任何值，直接在原模型上进行了修改。

    Raises:
        None: 没有引发任何异常。
    """

    # Embedding fusion
    W = model.gpt.embeddings.word_embeddings
    W_ = W.weight.detach().cast("float32")
    W_ = W_ - W_.mean(axis=-2, keepdim=True)
    paddle.assign(W_.cast(W.weight.dtype), output=W.weight)
    del W, W_
    layers = [layer for layer in model.gpt.decoder.layers]
    idx = 0
    for layer in layers:
        idx += 1
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(layer.norm1, [layer.self_attn.qkv_proj])
        fuse_ln_linear(layer.norm2, [layer.linear1])

        bake_mean_into_linear(layer.self_attn.out_proj)
        bake_mean_into_linear(layer.linear2)

        paddle.device.cuda.empty_cache()
        gc.collect()
    fuse_ln_linear(model.gpt.decoder.norm, [model.gpt.output_linear.out_linear])

    replace_modules(
        model,
        paddle.nn.LayerNorm,
        lambda _: RMSNorm(model.config.hidden_size),
        replace_layers=False,
    )


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    paddle.device.cuda.empty_cache()
    if device == "cuda":
        random_matrix = paddle.randn(size, size, dtype="float32").to("gpu")
    q, r = paddle.linalg.qr(random_matrix)
    q *= paddle.sign(paddle.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    """
    获取一个正交矩阵，可以是随机生成的、哈达马尔矩阵或者哈达马尔矩阵的FFN2版本。
    
    Args:
        size (int): 正交矩阵的大小。
        mode (str, optional): 生成方式，可选值为"random", "hadamard", "hadamard_ffn2"中的一种。默认为"random"。
            - "random"：生成一个随机正交矩阵。
            - "hadamard"：生成一个哈达马尔矩阵。
            - "hadamard_ffn2"：生成一个哈达马尔矩阵的FFN2版本。
        device (str, optional): 设备类型，默认为"cuda"。
    
    Returns:
        paddle.Tensor: 返回一个大小为size*size的正交矩阵，维度为2。
    
    Raises:
        ValueError: 如果mode不在"random", "hadamard", "hadamard_ffn2"中。
    """
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    elif mode == "hadamard_ffn2":
        return random_hadamard_matrix(size, device, True)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, Q):
    """
    旋转模型的词嵌入，使用Q矩阵进行变换。
    
    Args:
        model (paddlenlp.transformers.GPTForSequenceClassification): GPT模型对象。
        Q (Tensor, numpy.ndarray): 形状为[V, K]的矩阵，其中V是词典大小，K是新的词嵌入维度数。
            该函数会将原始的词嵌入矩阵W乘以Q得到新的词嵌入矩阵W'，然后更新模型的词嵌入。
    
    Returns:
        None, 无返回值，直接修改模型的词嵌入。
    """
    W = model.gpt.embeddings.word_embeddings
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32")
    paddle.assign(paddle.matmul(W_, Q).cast(dtype), output=W.weight)
    del W_, W


def rotate_head(model, Q) -> None:
    """
    对模型的head进行旋转操作。
    该函数会修改模型的参数。
    
    Args:
        model (paddlenlp.transformers.GPTForSequenceClassification): GPT模型实例。
        Q (Tensor, float32): 形状为[batch_size, seq_length, hidden_size]的Tensor，表示输入序列的特征向量。
    
    Returns:
        None, 无返回值，直接修改模型的参数。
    """
    # Rotate the head.
    W = model.gpt.output_linear.out_linear
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32").transpose((1, 0))
    paddle.assign(paddle.matmul(W_, Q).cast(dtype).transpose((1, 0)), output=W.weight)
    del W_, W


def rotate_attention_inputs(layer, Q) -> None:
    """
    旋转attention输入。
    
    Args:
        layer (nn.Layer): 自注意力层实例。
        Q (Tensor): 正交矩阵。
    
    Returns:
        None, 无返回值，直接修改了layer。
    """
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    W = layer.self_attn.qkv_proj

    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32").transpose((1, 0))
    paddle.assign(paddle.matmul(W_, Q).cast(dtype).transpose((1, 0)), output=W.weight)
    del W_, W


def rotate_attention_output(layer, Q) -> None:
    """
    对自注意力层的输出矩阵进行旋转。
    Args:
        layer (nn.Layer): 自注意力层，包含了self_attn属性。
        Q (Tensor): 旋转矩阵。
    
    Returns:
        None, 该函数不返回任何值。
        通过修改传入的layer中self_attn属性的weight和bias来实现对输出矩阵的旋转。
    """
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.out_proj
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32").transpose((1, 0))
    paddle.assign(paddle.matmul(Q.T, W_).transpose((1, 0)).cast(dtype), output=W.weight)
    if W.bias is not None:
        b = W.bias.detach().cast("float32")
        paddle.assign(paddle.matmul(Q.T, b).cast(dtype), output=W.bias)
    del W_, W


def rotate_mlp_input(layer, Q):
    """
    对MLP的输入进行旋转。
    
    Args:
        layer (nn.Layer): MLP层，其中包含一个线性层linear1。
        Q (Tensor): 旋转矩阵。
    
    Returns:
        None. 直接修改了layer.linear1.weight的值。
    
    Raises:
        None.
    """
    # Rotate the MLP input weights.
    W = layer.linear1
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32").transpose((1, 0))
    paddle.assign(paddle.matmul(W_, Q).cast(dtype).transpose((1, 0)), output=W.weight)
    del W_, W


def rotate_mlp_output(layer, Q):
    """
    对MLP输出进行旋转，旋转矩阵为Q。
    
    Args:
        layer (Layer): MLP层，包含线性层linear2和可选的偏置项bias。
        Q (Tensor, float32): 旋转矩阵。
    
    Returns:
        None, 该函数不返回任何值。
        通过修改传入的Mlp层的linear2的weight和bias来实现旋转操作。
    """
    # Rotate the MLP output weights and bias.
    W = layer.linear2
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32").transpose((1, 0))
    paddle.assign(paddle.matmul(Q.T, W_).cast(dtype).transpose((1, 0)), output=W.weight)
    if W.bias is not None:
        b = W.bias.detach().cast("float32")
        paddle.assign(paddle.matmul(Q.T, b).cast(dtype), output=W.bias)
    del W_, W


def rotate_ffn2(layer, Q):
    """
    对FFN2层进行旋转，将输入Q的列向量与线性层的权重矩阵相乘，并更新线性层的权重。
    该函数会修改传入的FFN2层的参数。
    
    Args:
        layer (nn.Layer): FFN2层。
        Q (Tensor): 旋转矩阵。
    
    Returns:
        None, 该函数不返回任何值，直接修改传入的FFN2层的参数。
    """
    W = layer.linear2
    dtype = W.weight.dtype
    W_ = W.weight.detach().cast("float32")
    paddle.assign(paddle.matmul(Q.T, W_).cast(dtype), output=W.weight)
    del W_, W


def apply_exact_had_to_linear(
    module, had_dim=-1, is_qkv=False, Q=None, hidden_size=None, config=None
):
    """
    旋转。
    
    Args:
        module (paddle.nn.Linear): Linear层对象，需要进行转换。
        had_dim (int, optional, default=-1): HAD维度。
        is_qkv (bool, optional, default=False): 是否是Q、K、V层，默认为False。
        Q (Optional[paddle.Tensor], optional): Q张量。
        hidden_size (Optional[int], optional): 隐藏层大小。
    
    Returns:
        None, 该函数不返回任何值。
    """

    W_ = module.weight
    dtype = W_.dtype
    W_ = W_.detach().cast("float32")
    if Q is not None:
        hadK = Q.cast("float32")
    if is_qkv:
        bias = module.bias.detach().cast("float32")
        group_size = config.num_attention_heads // config.num_key_value_heads
        had_dim = hadK.shape[0]
        bias = bias.reshape((-1, (group_size + 2) * had_dim))
        W_1 = W_.reshape((W_.shape[0], -1, (group_size + 2) * had_dim))
        q, k, v = paddle.split(
            W_1, num_or_sections=[group_size * had_dim, had_dim, had_dim], axis=-1
        )
        q_bias, k_bias, v_bias = paddle.split(
            bias, num_or_sections=[group_size * had_dim, had_dim, had_dim], axis=-1
        )
        transformed_v_bias = v_bias @ hadK
        transformed_v = (v.transpose((1, 0, 2)) @ hadK).transpose((1, 0, 2))
        temp = None
        bias_out = None
        for i in range(q.shape[1]):
            if temp is None:
                bias_out = paddle.concat(
                    [q_bias[i], k_bias[i], transformed_v_bias[i]], axis=0
                )
                temp = paddle.concat([q[:, i], k[:, i], transformed_v[:, i]], axis=1)
            else:
                temp = paddle.concat(
                    [temp, q[:, i], k[:, i], transformed_v[:, i]], axis=1
                )
                bias_out = paddle.concat(
                    [bias_out, q_bias[i], k_bias[i], transformed_v_bias[i]], axis=0
                )
        paddle.assign(temp.cast(dtype), output=module.weight)
        paddle.assign(bias_out.cast(module.bias.dtype), output=module.bias)
    else:
        init_shape = W_.shape
        W_ = W_.transpose((1, 0))
        temp = W_.reshape((init_shape[1], -1, had_dim))
        temp = temp @ hadK
        temp = temp.reshape((temp.shape[0], -1)).transpose((1, 0))
        paddle.assign(temp.cast(dtype), output=module.weight)


def rotate_ov_proj(layer, Q_head, head_dim, hidden_size, config):
    """
    对SelfAttention层的输出进行旋转。
    
    Args:
        layer (nn.Module): SelfAttention层，包含了qkv_proj和out_proj两个Linear层。
        Q_head (Tensor, optional): Q矩阵。
        head_dim (int): 每个头部的维度大小。
        hidden_size (int): 模型的隐藏状态维度大小。
    
    Returns:
        None. 直接修改了qkv_proj和out_proj的参数。
    """
    qkv_proj = layer.self_attn.qkv_proj
    o_proj = layer.self_attn.out_proj
    apply_exact_had_to_linear(
        qkv_proj, had_dim=head_dim, is_qkv=True, Q=Q_head, hidden_size=hidden_size, config=config
    )
    apply_exact_had_to_linear(
        o_proj, had_dim=head_dim, is_qkv=False, Q=Q_head, hidden_size=hidden_size, config=config
    )


def rotate_model(args, model, quarot_R1=True, quarot_ffn2=False):
    """
    对模型进行旋转操作，包括旋转embedding、旋转attention输入和输出以及旋转MLP的输入和输出。
    如果指定了`quarot_ffn2`为True，则还会对FFN2层进行旋转。
    
    Args:
        model (GPTForSequenceClassification): GPT模型，需要进行旋转操作。
        quarot_ffn2 (bool, optional): 是否对FFN2层进行旋转，默认为False。
            Defaults to False.
    
    Returns:
        None.
    """
    with paddle.no_grad():
        Q, _ = get_orthogonal_matrix(model.config.hidden_size, "hadamard")
        # Q_head = get_orthogonal_matrix(
        #    model.config.hidden_size // model.config.num_attention_heads, "hadamard"
        # )
        # config = model.config
        # num_heads = config.num_attention_heads
        # model_dim = config.hidden_size
        # head_dim = model_dim // num_heads

        if quarot_R1:
            rotate_embeddings(model, Q)
            paddle.device.cuda.empty_cache()
            rotate_head(model, Q)
            paddle.device.cuda.empty_cache()

        layers = [layer for layer in model.gpt.decoder.layers]
        Q_ffn2, block_size = get_orthogonal_matrix(
            layers[0].linear2.weight.shape[0], "hadamard_ffn2"
        )

        def pre_forward_hook(layer, input):
            with paddle.no_grad():
                transformed_input = paddle.matmul(input[0].cast("float32"), Q_ffn2)
            return transformed_input.cast(input[0].dtype)

        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
            if quarot_R1:
                rotate_attention_inputs(layers[idx], Q)
                rotate_attention_output(layers[idx], Q)
                rotate_mlp_input(layers[idx], Q)
                rotate_mlp_output(layers[idx], Q)
            if quarot_ffn2:
                rotate_ffn2(layers[idx], Q_ffn2.cast("float32"))
                layer.linear2.register_forward_pre_hook(pre_forward_hook)
            
            # rotate_ov_proj(layers[idx], Q_head, model.config.hidden_size//model.config.num_attention_heads, \
            #                 model.config.hidden_size, config)
            paddle.device.cuda.empty_cache()

        if dist.get_rank() == 0:
            config_path = os.path.join(args.save_path, "config.json")
            with open(config_path, "r") as fin:
                config = json.load(fin)
                if quarot_R1:
                    config["layernorm_only_std"] = True
                if quarot_ffn2:
                    config["ffn2_use_hardamard"] = True
                    config["hardamard_block_size"] = block_size
            with open(config_path, "w") as fout:
                json.dump(config, fout)
    return Q_ffn2, block_size
