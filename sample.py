


import numpy as np
import h5py
import scipy.io
mat = scipy.io.loadmat('annot.mat')
import torch

# print(mat["annot3"][0][0].shape)



x = torch.tensor (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
y = torch.tensor (([1, 2, 3]))

w = torch.tensor ([[1, 2, 3]])


y_hat = torch.matmul(x, w.t()).view(-1)

print(y_hat)



