# l4-pytorch
This is PyTorch implementation of
["L4: Practical loss-based stepsize adaptation for deep learning"](https://arxiv.org/abs/1802.05074)  By [Michal Rolínek](https://scholar.google.de/citations?user=DVdSTFQAAAAJ&hl=en), [Georg Martius](http://georg.playfulmachines.com/). 

---

**PLEASE NOTE:** 
This repository contains just the unedited `l4.py` file from the [original repository](https://github.com/iovdin/l4-pytorch), an empty `__init__.py` file to make it importable and no documentation whatsoever. 
For more information about *L4: Practical loss-based stepsize adaptation for deep learning*, please refer to the [original repository](https://github.com/iovdin/l4-pytorch) or the [original paper](https://arxiv.org/abs/1802.05074).

---

To install put ```l4.py``` to working directory


```python
from l4 import L4

#...

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)                                                           | create mode 100644 l4.py                                                                                                                  
# wrap original optimizer with L4
l4opt = L4(optimizer) 

#...

loss = F.nll_loss(output, target)
loss.backward() 

# Comment out original optimizer step
# optimizer.step()

# make step with L4 optimizer, dont forget to pass loss value
l4opt.step(loss)  
```

Tensorflow implementation can be found [here](https://github.com/martius-lab/l4-optimizer)
