import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import RowParallelLinear
from ...utils.log import logger

__all__ = ['ReorderFFNWeight']


class ReorderFFNWeight():
    def __init__(self,
                 model,
                 dtype='float16',
                 layer_prefix='mlp',
                 llama_ffn=False,
                 card_nums=8):

        '''
        1. find all (LN linear), and save their correspongding name in ffn1_ffn2_dict
            algo: forward once, record order, find each pair of (LN linear)
        2. add hook for all linears in ffn1_ffn2_dict
        3. in hook, sample abs max activation for each step and save it in scale_dict by linear name
        4. when sampling done, update ln weight and linear weight 
        '''
        self.model = model
        self.step = 0
        self.dtype = dtype
        self.layer_prefix = layer_prefix
        self.absmax_dict = {}
        self.reorder_index_dict = {}
        self.got_reorder_layers = False
        self.llama_ffn = llama_ffn
        self.print_step = 1
        self.ffn2_list = []
        self.card_nums = card_nums
        self.except_ffn2_list = []
        self.model.eval()
        self._apply_hook()

    def _get_reorder_layers(self):
        '''
        get all layers their weights need to be changed.
        only the layer norm and the linear after the layer norm.
        save them into a dict as a pair since they use one scale.
        '''
        except_ffn2_list = [l for l in self.except_ffn2_list if 'linear' in l]
        # print(except_ffn2_list)
        # print(self.ffn2_list)
        self.ffn1_ffn2_dict = {}
        for ffn2 in self.ffn2_list:
            if self.llama_ffn:
                lineark=int(ffn2.split('_')[-1])
                if 'linear_'+str(lineark-1) and 'linear_'+str(lineark+1) in except_ffn2_list:
                    self.ffn1_ffn2_dict['linear_'+str(lineark-1)]=ffn2
                    self.ffn1_ffn2_dict['linear_'+str(lineark+1)]=ffn2
            else:
                lineark = int(ffn2.split('_')[-1])
                if 'linear_' + str(lineark - 1) in except_ffn2_list:
                    self.ffn1_ffn2_dict['linear_' + str(lineark - 1)] = ffn2
        print("self.ffn1_ffn2_dict: ", self.ffn1_ffn2_dict)
        self.got_reorder_layers = True

    def _forward_ffn2_pre_hook(self, layer, input):
        if self.step == 0 and layer.full_name() in self.ffn2_list:
            self.step += 1
        if self.step == 0:
            self.ffn2_list.append(layer.full_name())
        if self.step == 1:
            if self.got_reorder_layers == False:
                self._get_reorder_layers()
        if self.step > 0 and layer.full_name() in self.ffn1_ffn2_dict.values():
            self._sample_absmax(input, layer)
        return input
    
    def _sample_absmax(self, input, layer):
        # print("---------layer-----------",layer)
        # print("----------input-------------",input)
        ln_name = layer.full_name()
        x = input[0] if type(input) == tuple else input
        x.stop_gradient = True
        x = x.cast('float32')
        reduce_axis = (0, 1) if len(x.shape) > 2 else 0
        abs_max_values = paddle.max(paddle.abs(x), axis=reduce_axis) # shape [896]

        # abs_max_values = paddle.where(abs_max_values == np.float32(0.0),
        #                               np.float32(1e-8), abs_max_values)
        # print("abs_max_values",abs_max_values)
        if ln_name not in self.absmax_dict:
            self.absmax_dict[ln_name] = abs_max_values
        else:
            self.absmax_dict[ln_name] = paddle.maximum(abs_max_values, self.absmax_dict[ln_name])
        
        # if self.print_step == self.step:
        #     print('[reorder] Step [{}]: {}. abs_max: {}'.format(self.step, ln_name, round(float(self.absmax_dict[ln_name].max()), 5)))
        #     self.print_step += 1
        
        # if ln_name == "linear_319":
        #     print("---------------> x: ", float(x.max()), x.shape, abs_max_values.shape, float(abs_max_values.max()))

    def _apply_hook(self):
        self._forward_hook_list = []
        for layer_name, sub_layer in self.model.named_sublayers():
            if type(sub_layer) in [RowParallelLinear] and self.layer_prefix in layer_name:
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(self._forward_ffn2_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)
            elif self.layer_prefix in layer_name: 
                self.except_ffn2_list.append(sub_layer.full_name())

    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()

    def update_weight(self):
        '''
        update weight of smooth layers.
        firstly compute s and update linear's weight,
        then update LN's weight by corresponding linear and s
        '''
        # get reorderindex
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.ffn1_ffn2_dict.values():
                xmax_percard = self.absmax_dict[layer_name]
                xmax = paddle.distributed.collective._c_concat(
                    xmax_percard, group=paddle.distributed.get_group())
                if self.card_nums is not None:
                    self.reorder_index_dict[layer_name], _ = tensor_calc_reorder_index(xmax, self.card_nums)

        # update ffn2 weight
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.ffn1_ffn2_dict.values():
                reorder_index = self.reorder_index_dict[layer_name]
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'w_0' in param.name:
                        new_weight = reorder_weight(param, reorder_dim=0, reorder_index=reorder_index)
                        paddle.assign(new_weight.cast(self.dtype), output=param)
                        break
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'b_0' in param.name:
                        new_bias = reorder_bias(param, reorder_dim=0, reorder_index=reorder_index)
                        paddle.assign(new_bias.cast(self.dtype), output=param)
                        break

        # update ffn1 weight
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if self.ffn1_ffn2_dict.__contains__(layer_name):
                ffn2_name = self.ffn1_ffn2_dict[layer_name]
                reorder_index = self.reorder_index_dict[ffn2_name]
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'w_0' in param.name:
                        new_weight = reorder_weight(param, reorder_dim=1, reorder_index=reorder_index)
                        paddle.assign(new_weight.cast(self.dtype), output=param)
                        break
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'b_0' in param.name:
                        new_bias = reorder_bias(param, reorder_dim=1, reorder_index=reorder_index)
                        paddle.assign(new_bias.cast(self.dtype), output=param)
                        break

        self._remove_hook()
        paddle.device.cuda.empty_cache()

def tensor_calc_reorder_index(xmax, n_clusters):
    all_counts = []
    card_nums = n_clusters
    all_index = paddle.argsort(xmax, axis=-1)
    couts = xmax.shape[0] / card_nums
    for i in range(card_nums):
        all_counts.append(couts)
        i+=1
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def reorder_weight(inputs, reorder_dim, reorder_index):
    if reorder_dim == 0:
        inputs_t = inputs.t()
        inputs_all = paddle.distributed.collective._c_concat(
                inputs_t, group=paddle.distributed.get_group())
        inputs_all_t = inputs_all.t()
        # print('reorder_dim==0')
        # print(inputs_all_t)
        if  reorder_index is not None:
                inputs_all_t = paddle.index_select(inputs_all_t, reorder_index,(reorder_dim))
        # print(inputs_all_t)
        inputs_all=inputs_all_t.t()
        inputs = paddle.distributed.collective._c_split(
                    inputs_all, group=paddle.distributed.get_group())
        inputs=inputs.t()
    else:
        ffn_fc1, gate = paddle.chunk(inputs, chunks=2, axis=-1)
        ffn_fc1_all = paddle.distributed.collective._c_concat(
                ffn_fc1, group=paddle.distributed.get_group())
        gate_all = paddle.distributed.collective._c_concat(
                gate, group=paddle.distributed.get_group())
        if reorder_index is not None:
            ffn_fc1_all = paddle.index_select(ffn_fc1_all, reorder_index,(reorder_dim))
            gate_all = paddle.index_select(gate_all, reorder_index,(reorder_dim))
        ffn_fc1_new = paddle.distributed.collective._c_split(
                    ffn_fc1_all, group=paddle.distributed.get_group())
        gate_new = paddle.distributed.collective._c_split(
                    gate_all, group=paddle.distributed.get_group())
        inputs = paddle.concat([ffn_fc1_new, gate_new], axis=-1)
    return inputs

def reorder_bias(inputs, reorder_dim, reorder_index):
    if reorder_dim != 0:
        ffn_fc1, gate = paddle.chunk(inputs, chunks=2, axis=-1)
        ffn_fc1_all = paddle.distributed.collective._c_concat(
                ffn_fc1, group=paddle.distributed.get_group())
        gate_all = paddle.distributed.collective._c_concat(
                gate, group=paddle.distributed.get_group())
        if reorder_index is not None:
            ffn_fc1_all = paddle.index_select(ffn_fc1_all, reorder_index,(0))
            gate_all = paddle.index_select(gate_all, reorder_index,(0))
        ffn_fc1_new = paddle.distributed.collective._c_split(
                    ffn_fc1_all, group=paddle.distributed.get_group())
        gate_new = paddle.distributed.collective._c_split(
                    gate_all, group=paddle.distributed.get_group())
        inputs = paddle.concat([ffn_fc1_new, gate_new], axis=-1)
    return inputs(base)