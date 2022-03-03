from os import scandir
import torch
from torch.autograd import Variable

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    squre = torch.square(vectors)
    if torch.min(squre) < 0:
        print("(--------------------------------------------)")
    s_squared_norm = torch.sum(squre, axis, keepdims=True)
    # print("Norm - ",s_squared_norm)
    k = ((torch.sqrt(s_squared_norm.float())))
    # print("squre root - ",k)
    # scale = s_squared_norm / (1 + s_squared_norm)/ torch.sqrt(s_squared_norm + 1e-07)
    scale = ((s_squared_norm/(1+s_squared_norm))/torch.sqrt(s_squared_norm + 1e-07))
    # if torch.min(scale)<0:
    # print("scale - ",scale)    
    output = scale * vectors
    # if torch.sum(output) < 0:
    #     # print("Norm - ",s_squared_norm)
    #     # print("scale - ",scale)   
    #     print(k)
    #     print("summation - ", torch.sum(output))
    #     print("output - ", output)

    return output
#     s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
#     scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
#     return scale * vectors

def squash_fn(input, dim=-1):
    norm = input.norm(p=2, dim=dim, keepdim=True)
    scale = norm / (1 + norm ** 2)
    return scale * input
def squash_d(x, axis=-1):
    s_squared_norm = torch.sum(torch.square(x), axis, keepdims=True)
    scale = torch.sqrt(s_squared_norm +1e-07)
    return x / scale

# import numpy as np

# vectors = np.array([3,1,1,0,2])
# tensor = torch.from_numpy(vectors)

# output = squash(tensor)
# output = squash(output)
# output = squash(output)
# output = squash(output)
# print(output)