import gc
import paddle
import paddle.distributed.fleet as fleet
from .utils import find_parent_layer_and_sub_name
from ...utils.log import logger
__all__ = ['PermRotate']

class PermRotate():
    """ 
    Zigzag Permutation is a method to balance the outliers’ magnitudes among various blocks.
    (Ref: DUQUANT https://arxiv.org/pdf/2406.01721)
    We permutate the activations of linear2, then use a rotation to smooth outliers.
    """
    def __init__(
        self,
        model,
        dp_degree=1,
    ):
        """
        Args:
        model(paddle.nn.Layer, required): the model to be permutated.
        dp_degree (int, optional): The degree of data parallelism. Defaults to 1.
        """
        self.model = model
        self.model.eval()
        self.sampled_inputs = {}
        self.permutation_dict = {}
        self.dp_degree = dp_degree
        try:
            hcg = fleet.get_hybrid_communicate_group()
            self.mp_degree = hcg.get_model_parallel_world_size()
            self.mp_id = hcg.get_model_parallel_rank()
            self.dp_id = hcg.get_data_parallel_rank()
        except:
            self.mp_degree = 1
            self.mp_id = 0
            self.dp_id = 0
        self._apply_hook()

    def _apply_hook(self):
        """
        Register forward pre-hook on all layers with name `linear2`.
        """
        self._forward_hook_list = []
        for cur_name, sub_layer in self.model.named_sublayers():
            if "linear2" in cur_name:
                logger.debug(f"Apply hook: cur_name: {cur_name}, sub_layer.full_name(): {sub_layer.full_name()}")
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)
    
    def _forward_pre_hook(self, layer, input):
        """
        Apply sample_act in forward pre hook.
        """
        self._sample_act(input, layer.full_name())
        return input

    def _sample_act(self, input, layer_name):
        """
        Collect the maximum absolute value of each sample and store them into dict.
        """
        x = input[0].cast("float32") if type(input) == tuple else input.cast("float32")
        x_dtype = x.dtype
        x.stop_gradient = True
        act_max = x.squeeze(0).abs().max(axis=0, keepdim=True)
        if layer_name not in self.sampled_inputs.keys():
            self.sampled_inputs[layer_name] = act_max
        else:
            self.sampled_inputs[layer_name] = paddle.concat([act_max, self.sampled_inputs[layer_name]], axis=0)

    def get_permutation_zigzag(self, block_size=512):
        """
        Get zigzag permutation from abs max activation values.
        """
        for layer_name, sampled_input in self.sampled_inputs.items():
            if self.dp_degree > 1:
                sampled_input_list = []
                paddle.distributed.all_gather(sampled_input_list, sampled_input)
                sampled_input = paddle.concat([sampled_input_list[k * self.mp_degree + self.mp_id] \
                          for k in range(self.dp_degree)], axis=0)
            hidden_dim = sampled_input.shape[-1]
            sampled_input = paddle.max(sampled_input, axis=0, keepdim=True).squeeze(0)
            pairs = [(i, sampled_input[i].item()) for i in range(hidden_dim)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = self._zigzag(pairs, hidden_dim, block_size)
            for i in range(len(pairs)):
                pairs[i].sort(key=lambda x: x[1], reverse=True)
            
            perm = paddle.zeros(hidden_dim, dtype='int32')
            for i in range(len(pairs)):
                perm[i * block_size:(i + 1) * block_size] = paddle.to_tensor([_[0] for _ in pairs[i]], dtype='int32')
            self.permutation_dict[layer_name] = perm

    def permutation(self):
        """
        Permutate the activations of linear2 by reorder the weights of linear1.
        Permutate the weights of linear2 to ensure the computation invariance.
        """
        self.get_permutation_zigzag(block_size=512)
        for cur_name, sub_layer in self.model.named_sublayers():
            if "linear2" in cur_name:
                perm = self.permutation_dict[sub_layer.full_name()]
                weight_tmp = sub_layer.weight.detach()[perm, :]
                paddle.assign(weight_tmp, sub_layer.weight)
                del weight_tmp
                gc.collect()
            elif 'linear1' in cur_name:
                layer_num = int(sub_layer.full_name().split('_')[-1])
                perm = self.permutation_dict["linear_" + str(layer_num + 1)]
                perm_tmp = paddle.zeros(sub_layer.weight.shape[-1], dtype='int32')
                for i in range(len(perm)):
                    perm_tmp[2 * i] = perm[i] * 2
                    perm_tmp[2 * i + 1] = perm[i] * 2 + 1
                weight_tmp = sub_layer.weight.detach()[:, perm_tmp]
                paddle.assign(weight_tmp, sub_layer.weight)
                bias_tmp = sub_layer.bias.detach()[perm_tmp]
                paddle.assign(bias_tmp, sub_layer.bias)
                del weight_tmp, bias_tmp
                gc.collect()
        self._remove_hook()
        paddle.device.cuda.empty_cache()

    def _zigzag(self, numbers, hidden_dim, block_size=512):
        """
        Zigzag permutation algorithm.
        """
        cur = 0
        up = True
        l = [[] for i in range(hidden_dim // block_size)]
        for i in range(len(numbers)):
            l[cur].append(numbers[i])
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

    def _remove_hook(self):
        """
        Remove all hooks registered.
        """
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []
