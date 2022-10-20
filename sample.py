


import numpy as np
import h5py
import scipy.io
mat = scipy.io.loadmat('/media/pranoy/Pranoy/mpi_inf_3dhp/S1/Seq1/annot.mat')
import torch

print(len(mat["annot3"][0]))

print



# x = torch.tensor (([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
# y = torch.tensor (([1, 2, 3]))

# w = torch.tensor ([[1, 2, 3]])


# y_hat = torch.matmul(x, w.t()).view(-1)

# print(y_hat)



