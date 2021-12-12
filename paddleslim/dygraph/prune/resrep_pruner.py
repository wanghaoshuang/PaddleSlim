import logging
import copy
import numpy as np
import paddle
import time
from paddle.fluid import core
from paddleslim.common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner
from paddleslim.analysis import dygraph_flops as flops
from .var_group import DygraphPruningCollections
from paddleslim.core import GraphWrapper, dygraph2program

__all__ = ['ResRepPruner']

_logger = get_logger(__name__, logging.INFO)


def find_parent_layer_and_sub_name(model, name):
    """
    Given the model and the name of a layer, find the parent layer and
    the sub_name of the layer.
    For example, if name is 'block_1/convbn_1/conv_1', the parent layer is
    'block_1/convbn_1' and the sub_name is `conv_1`.
    Args:
        model(paddle.nn.Layer): the model to be quantized.
        name(string): the name of a layer
    Returns:
        parent_layer, subname
    """
    assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."
    assert len(name) > 0, "The input (name) should not be empty."

    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == '.':
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name

class BatchNormWrapper(paddle.nn.Layer):
    def __init__(self,
                 layer):
        super(BatchNormWrapper, self).__init__()
        self._layer = layer
        compactor_name = self._layer.full_name() + ".compactor.weight"
        mask_name = self._layer.full_name() + ".compactor.mask"
        self._channel = self._layer.weight.shape[0]
        initializer = paddle.nn.initializer.Assign(np.identity(self._channel).reshape([self._channel, self._channel, 1, 1]))
        compactor_attr  = paddle.ParamAttr(name=compactor_name,
                                          initializer=initializer,
                                          learning_rate=1.0,
                                          regularizer=paddle.regularizer.L2Decay(0.),
                                          #regularizer=None,
                                          trainable=True,
                                          do_model_average=False,
                                          need_clip=False)
        self.compactor = paddle.nn.Conv2D(self._channel, self._channel, 1,
                                          bias_attr=False,
                                          weight_attr=compactor_attr)
        self.mask = np.ones([self._channel,]).astype('float32')

    def forward(self, *inputs, **kwargs):
        out = self._layer(*inputs, **kwargs)
        return self.compactor(out)


class ResRepPruner():
    def __init__(self, model, inputs, dtypes, opt, target_flops):
        self.model = model
        self.opt = opt
        self.target_flops = target_flops

        self.original_flops, self.param2flops = flops(model, inputs, dtypes=dtypes, only_conv=True, detail=True)

        self.conv2partners = {} # {"conv2d_0.weight": [("conv2d_1.weight", 1, (c_out, c_in, k, k), op)]}
        self.brothers = {}
        self.collections = DygraphPruningCollections(model, inputs)
        for collection in self.collections:
            _name = collection.master_name
            _axis = collection.master_axis
            assert _axis == 0
            if _name not in self.conv2partners:
                self.conv2partners[_name] = []
            for detail in collection.all_pruning_details():
                self.conv2partners[_name].append((detail.name, detail.axis, detail.var.shape(), detail.op))
                if detail.axis == 0 and detail.op.type()=="conv2d":
                    if _name not in self.brothers:
                        self.brothers[_name] = []
                    self.brothers[_name].append(detail.name)
            
        program = dygraph2program(model, inputs=inputs)
        graph = GraphWrapper(program) 

        self.conv2bn = {}
        self.bn2conv = {}
        self.conv2shape = {}
        for op in graph.ops():
            if op.type() == "conv2d":
               _filter = op.inputs("Filter")[0]
               conv_name = _filter.name()
               conv_shape = _filter.shape
               self.conv2shape[conv_name] = conv_shape 
               for op in graph.next_ops(op):
                   if op.type() == "batch_norm":
                       bn_name = op.inputs("Scale")[0].name()
                       self.conv2bn[conv_name] = bn_name
                       self.bn2conv[bn_name] = conv_name
                       
        self.convs = [] # global channel id --> conv2d name
        self.offsets = [] # global channel id --> channel offset in conv
        self.conv2compactor = {}
        self.conv2mask = {}
        self.total_channels = 0
        for _name, _layer in self.model.named_sublayers():
            if isinstance(_layer, (paddle.nn.layer.norm.BatchNorm2D, paddle.fluid.dygraph.nn.BatchNorm)):
                parent_layer, sub_name = \
                    find_parent_layer_and_sub_name(self.model, _name)

                bn = BatchNormWrapper(_layer)

                setattr(parent_layer, sub_name, bn)
                conv_name = self.bn2conv[bn._layer.weight.name]
                self.conv2compactor[conv_name] = bn.compactor
                self.conv2mask[conv_name] = bn.mask
                self.total_channels += len(bn.mask)
                if conv_name in self.conv2partners:
                    self.convs.extend([conv_name] * len(bn.mask))
                    self.offsets.extend(range(len(bn.mask)))
                
        self.cur_flops = self.original_flops
        self.current_step = 0
        self.cur_deactivated = 0    

        batch_size = 256
        self.warmup_steps =  5*1281167 // batch_size
        self.interval_steps = 200
        self.deactivated_granularity = 4
        self.least_channel = 1
        self.lasso_strength = 1e-4

    def reset_mask(self):
        for _conv in self.conv2mask:
            self.conv2mask[_conv][:] = 1

    def channel_scores(self, weight):
        weight = weight.numpy()
        #ele_count = np.product(weight.shape[1:])
        return np.sqrt(np.sum(weight**2, axis=(1,2,3)))
        # return np.abs(weight).mean(axis=(1,2,3)) # l1norm

    def update_mask(self):
        self.reset_mask()
        if self.cur_flops < self.target_flops * self.original_flops:
            return
        self.cur_deactivated = self.cur_deactivated + self.deactivated_granularity
        scores = []
        for _conv, _brothers in self.brothers.items():
            _scores = self.channel_scores(self.conv2compactor[_conv].weight)
            for _brother in _brothers:
                _scores += self.channel_scores(self.conv2compactor[_brother].weight)
            scores.extend(list(_scores))
        idxes = np.argsort(scores)
        deactivated_count = 0
        param2flops = copy.deepcopy(self.param2flops)
        self.cur_flops = self.original_flops
        for i in idxes:
            if self.cur_flops < self.target_flops * self.original_flops:
                break
            conv = self.convs[i]
            mask = self.conv2mask[conv]
            if np.sum(mask) <= self.least_channel:
                continue
            if deactivated_count >= self.cur_deactivated:
                break

            one_channel_flops = param2flops[conv] / len(mask)            
            self.cur_flops -= one_channel_flops
            param2flops[conv] -= one_channel_flops

            mask[self.offsets[i]] = 0
            deactivated_count += 1

            for _name, _axis, _shape, _op in self.conv2partners[conv]:
                if len(_shape) == 4 and _op.type()=="conv2d":
                    if _axis == 0: # brothers
                        _mask = self.conv2mask[_name]
                        deactivated_count += 1
                        one_channel_flops = param2flops[_name] / len(_mask)
                        self.cur_flops -= one_channel_flops
                        param2flops[_name] -= one_channel_flops
                    elif _axis == 1: # successors
                        assert _shape[_axis] == len(mask)
                        one_channel_flops = param2flops[_name] / _shape[_axis]
                        self.cur_flops -= one_channel_flops
                        param2flops[_name] -= one_channel_flops
        tmp = [(_p, _f / self.param2flops[_p], _f) for _p, _f in  param2flops.items()]
        print(f"self.cur_flops: {self.cur_flops/self.original_flops}; deactivated_count: {deactivated_count}; {tmp}")
    def mask_compactor_grad(self):
        for _conv, _compactor in self.conv2compactor.items():
           _compactor_grad = _compactor.weight.grad.value().get_tensor()
           _mask = self.conv2mask[_conv].reshape([-1, 1, 1, 1])
           lasso_grad = _compactor.weight * (paddle.sum((_compactor.weight ** 2), axis=(1, 2, 3), keepdim=True) ** (-0.5))
           _grad = _compactor.weight.grad * paddle.to_tensor(_mask) + lasso_grad * self.lasso_strength 

           p = _compactor_grad._place()
           if p.is_cpu_place():
               place = paddle.CPUPlace()
           elif p.is_cuda_pinned_place():
               place = paddle.CUDAPinnedPlace()
           else:
               p = core.Place()
               p.set_place(_compactor_grad._place())
               place = paddle.CUDAPlace(p.gpu_device_id())
           _compactor.weight._set_grad_ivar(_grad)
           return
           _compactor_grad.set(_grad, place)
       

    def step(self):
        self.opt.step()
        if (self.current_step > self.warmup_steps):
            if ((self.current_step-self.warmup_steps) % self.interval_steps == 0):
                self.update_mask()
                print(f"step: {self.current_step}; self.cur_flops: {self.cur_flops/self.original_flops}; deactivate channels: {self.cur_deactivated}/{self.total_channels};")
            self.mask_compactor_grad()
        self.current_step += 1

    def clear_grad(self):
        self.opt.clear_grad()

    def minimize(self, loss):
        self.opt.minimize(loss)

    def state_dict(self):
        return self.opt.state_dict()

    def set_state_dict(self, state_dict):
        self.opt.set_state_dict(state_dict)
