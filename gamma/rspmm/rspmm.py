import os
import sys

import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]


class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


def _create_rspmm_function(sum_op, mul_op):
    class GeneralizedRSPMMFunction(autograd.Function):
        @staticmethod
        def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
            node_in, node_out = edge_index
            key = node_in * (node_out.max() + 1) + node_out
            assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

            func_name = f"rspmm_{sum_op}_{mul_op}_forward"
            if input.device.type == "cuda":
                forward_fn = getattr(rspmm, func_name + "_cuda")
            else:
                forward_fn = getattr(rspmm, func_name + "_cpu")

            output = forward_fn(edge_index, edge_type, edge_weight, relation, input)
            ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
            ctx.sum_op = sum_op
            ctx.mul_op = mul_op
            return output

        @staticmethod
        def backward(ctx, output_grad):
            func_name = f"rspmm_{ctx.sum_op}_{ctx.mul_op}_backward"
            if output_grad.device.type == "cuda":
                backward_fn = getattr(rspmm, func_name + "_cuda")
            else:
                backward_fn = getattr(rspmm, func_name + "_cpu")

            weight_grad, relation_grad, input_grad = backward_fn(*ctx.saved_tensors, output_grad)
            return None, None, weight_grad, relation_grad, input_grad

    name = f"RSPMM{sum_op.capitalize()}{''.join(word.capitalize() for word in mul_op.split('_'))}Function"
    GeneralizedRSPMMFunction.__name__ = name
    return GeneralizedRSPMMFunction


def generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    class_mul = "".join(word.capitalize() for word in mul.split('_'))
    name = "RSPMM%s%sFunction" % (sum.capitalize(), class_mul)
    if not hasattr(module, name):
        raise ValueError(
            "No generalized rspmm implementation found for summation `%s` and multiplication `%s`" % (sum, mul))
    Function = getattr(module, name)

    edge_index_swapped = torch.stack([edge_index[1], edge_index[0]], dim=0)

    node_in, node_out = edge_index_swapped
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()

    return Function.apply(edge_index_swapped[:, order], edge_type[order], edge_weight[order], relation, input)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available() and not sys.platform.startswith('darwin'):
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return cpp_extension.load(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


print("Load rspmm extension. This may take a while...")
path = os.path.join(os.path.dirname(__file__), "source")
rspmm = load_extension("rspmm", [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")])

for _sum in ["add", "min", "max"]:
    for _mul in ["complex", "split_complex", "dual"]:
        setattr(module, f"RSPMM{_sum.capitalize()}{''.join(word.capitalize() for word in _mul.split('_'))}Function",
                _create_rspmm_function(_sum, _mul))
