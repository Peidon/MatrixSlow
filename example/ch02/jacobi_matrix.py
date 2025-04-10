import torch
from torch.autograd import Variable


# 根据输入输出的维度构建一个jacobian空矩阵，[[输入0展开，输出展开], [输入1展开，输出展开]...]
def make_jacobian(input, num_out):
    if torch.is_tensor(input) or isinstance(input, Variable):
        return torch.zeros(input.nelement(), num_out)
    else:
        return type(input)(filter(lambda x: x is not None,
                                  (make_jacobian(elem, num_out) for elem in input)))


# 辅助函数，将tensor值迭代输出
def iter_tensors(x, only_requiring_grad=False):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Variable):
        if x.requires_grad or not only_requiring_grad:
            yield x.data
    else:
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


# 将tensor变为连续值
def contiguous(input):
    if torch.is_tensor(input):
        return input.contiguous()
    elif isinstance(input, Variable):
        return input.contiguous()
    else:
        return type(input)(contiguous(e) for e in input)


# 构建jacobian矩阵
def get_jacobian(fn, input, target):
    perturbation = 1e-6
    input = contiguous(input)
    output_size = fn(*input).numel()
    jacobian = make_jacobian(target, output_size)

    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    outa = torch.DoubleTensor(output_size)  # 保存后向偏移值
    outb = torch.DoubleTensor(output_size)  # 保存前向偏移值

    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1)
        for i in range(flat_tensor.nelement()):
            orig = flat_tensor[i].clone()  # 进行clone操作，不进行原地修改
            flat_tensor[i] = orig - perturbation
            outa.copy_(fn(*input).view(-1))
            flat_tensor[i] = orig + perturbation
            outb.copy_(fn(*input).view(-1))
            flat_tensor[i] = orig

            d_tensor[i] = (outb - outa) / (2 * perturbation)  # 中心差分计算

    return jacobian


def func(data1, data2):
    return torch.matmul(data1, data2)


if __name__ == "__main__":
    x0 = torch.tensor([[0, 1], [2, 3], [4, 5]], requires_grad=False, dtype=torch.double)
    x1 = torch.tensor([[6, 7], [8, 9]], requires_grad=False, dtype=torch.double)
    inputs = (x0, x1)
    y1 = func(*inputs)
    J = get_jacobian(func, inputs, inputs)
    J_x0 = J[0]
    J_x1 = J[1]
    print(J_x1.T)
