# Tensor Introduction

## Contents

1. Tensors
2. Interaction between tensors
3. autograd

---

## Tensor 101

- Pytorch structure to work with variables -> pytorch tensors
- similar to numpy arrays, but more powerful
- because tensors can automacially calculate gradients
- information about dependencies to other tensors

### automatic gradience

```
# create a tensor with gradients enabled
x = torch.tensor(1.0, requires_grad=True)

# create second tensor depending on first tensor
y = (x-3) * (x-6) * (x-4)

# caculate gradients
y.backward()

# show gradient of first tensor
print(x.grad) ## results: tensor(31.)
```
