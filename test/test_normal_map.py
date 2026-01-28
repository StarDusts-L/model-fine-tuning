import math

import torch
from torch.distributions import Normal

d = Normal(0, math.sqrt(1))
def create_normal_map(off_set = 0.9677083,size = 16):
    start = 1 - off_set
    spare = (off_set - start)/14
    while True:
        print(d.icdf(torch.tensor([start])).item()/1.8481)
        start += spare
        if start >=off_set:
            break
create_normal_map()