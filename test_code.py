
import torch
from torch import nn

a = torch.cuda.FloatTensor([1., 2.])


sq = nn.Sequential(
         nn.Linear(20, 20),
         nn.ReLU(),
         nn.Linear(20, 4),
         nn.Softmax()
)

model = sq.cuda()

print(model.eval())

