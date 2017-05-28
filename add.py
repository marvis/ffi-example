import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import add_lib

class FuncAdd(Function):
    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            add_lib.add_forward(input1, input2, output)
        else:
            add_lib.add_forward_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            add_lib.add_backward(grad_output, grad_input)
        else:
            add_lib.add_backward_cuda(grad_output, grad_input)
        return grad_input

class NNAdd(Module):
    def forward(self, input1, input2):
        return FuncAdd()(input1, input2)
