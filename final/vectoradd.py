import numpy as np
import torch

device=torch.device('cuda')
A=torch.tensor([0,1,2,3,4,5,6,7,8,9],device=device)
B=torch.tensor([9,8,7,6,5,4,3,2,1,0],device=device)

C=A+B;

for i in range(10):
    print(C[i])
