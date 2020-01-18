import runmfcc
from time import time
import numpy as np

a = runmfcc.initialize()
l = np.random.randint(200, size=2000).tolist()
print(l)
x = a.runmfcc(l)
print(x)

a.exit()
