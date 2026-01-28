import torch

a = torch.tensor([[1/13,2/13,3/13,4/13,5/13],
                  [1/13,2/13,3/13,4/13,5/13],
                  [1/13,3/13,3/13,4/13,5/13]],dtype=torch.float)
print(torch.linalg.matrix_rank(a))
print(torch.linalg.norm(a))
print(torch.linalg.matrix_power(a))