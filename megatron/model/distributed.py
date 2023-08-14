# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import mpu
from .module import MegatronModule

import msamp
from msamp.common.tensor import ScalingMeta, ScalingTensor
from msamp.common.dtype import Dtypes

wgrad_qtype = Dtypes.kfloat8_e4m3
wgrad_dtype = torch.fp8e4m3

class MemoryBuffer:

    def __init__(self, numel, numel_padded, dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to store and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffer_param_index_map = {}
            data_parallel_world_size = mpu.get_data_parallel_world_size()

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            fp8_params = []
            for param in self.module.parameters():
                if param.requires_grad:
                    if torch.is_tensor(param):
                        dtype = _get_buffer_type(param)
                        type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()
                    else:
                        fp8_params.append(param)

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    if not torch.is_tensor(param):
                        continue
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])
                    if dtype not in self._grad_buffer_param_index_map:
                        self._grad_buffer_param_index_map[dtype] = {}
                    self._grad_buffer_param_index_map[dtype][param] = (
                        type_num_elements[dtype],
                        type_num_elements[dtype] + param.data.nelement(),
                    )

            self._grad_buffer_num_params = [0 for _ in range(data_parallel_world_size)]
            if len(fp8_params) > 0:
                self._grad_buffer_param_index_map[wgrad_dtype] = {}
                fp8_params_with_size = [(p, (-p.numel(), i % data_parallel_world_size)) for i, p in enumerate(fp8_params)]
                fp8_params_with_size.sort(key=lambda e: e[1])
                fp8_mems = [0 for _ in range(data_parallel_world_size)]
                fp8_partitions = [[] for _ in range(data_parallel_world_size)]
                for p, _ in fp8_params_with_size:
                    target_rank = fp8_mems.index(min(fp8_mems))
                    fp8_mems[target_rank] += p.numel()
                    fp8_partitions[target_rank].append(p)
                    self._grad_buffer_num_params[target_rank] += 1
                max_fp8_mems = max(fp8_mems)
                num_elements = max_fp8_mems * data_parallel_world_size
                assert wgrad_dtype not in self._grad_buffers
                
                # get dtype
                dtype = wgrad_dtype
                self._grad_buffers[wgrad_dtype] = MemoryBuffer(num_elements, num_elements, dtype)
                num_params = len(fp8_params)
                window_size = 1
                scales = torch.ones((num_params,), device='cuda')
                scale_invs = torch.ones((num_params,), device='cuda')
                amaxs = torch.zeros((num_params, window_size), device='cuda')
                scaling_grads = []
                t = 0
                pre_scale = 1.0 / math.sqrt(data_parallel_world_size)
                for pi in range(data_parallel_world_size):
                    start = pi * max_fp8_mems
                    for p in fp8_partitions[pi]:
                        meta = ScalingMeta(wgrad_qtype, scale=scales[t],
                                               scale_inv=scale_invs[t], amax=amaxs[t])
                        meta.pre_scale = pre_scale
                        t += 1
                        p.main_grad = ScalingTensor(self._grad_buffers[wgrad_dtype].get(p.shape, start), meta)
                        self._grad_buffer_param_index_map[wgrad_dtype][p] = (start, start + p.numel())
                        start += p.numel()
                        scaling_grads.append(p.main_grad)
                    assert start <= num_elements
                self._fp8_main_grad_scales = scales
                self._fp8_main_grad_scale_invs = scale_invs
                self._fp8_main_grad_amaxs = amaxs
                self._scaling_grads = scaling_grads

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    if torch.is_tensor(param):
                        # Expand so we get access to grad_fn.
                        param_tmp = param.expand_as(param)
                        # Get the gradient accumulator functtion.
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        grad_acc.register_hook(self._make_param_hook(param))
                    else:
                        hook = self._fp8_make_param_hook(param)
                        grad_acc = param.register_backward_post_hook(hook)
                    self.grad_accs.append(grad_acc)

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook

    def _fp8_make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                param.main_grad.copy_((param.main_grad.to(param.grad.dtype) + \
                    param.grad).cast(wgrad_qtype, meta=param.main_grad.meta))
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()


    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())


    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(buffer_.data, group=mpu.get_data_parallel_group())
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
