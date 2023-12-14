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

- simple network
  - input x is used to calculate y, which is used to caculate z
  - x --> y --> z (forward pass; the link!)
  - x <-- y <-- z (backpropagation; needs calculating derivatives)
    dy/dx dz/dy

-> Change of z based on change of x: dz/dx = dz/dy \* dy/dx (Chain rule)

그럼 weights 는??

- update of weights

  - calculated output z
  - True output : t
  - Error E = (z-t)\*\*2
  - Weights can be considered as nodes as well!
  - z = f(y, w2)
  - Optimizer update weights based on gradients

- More complex network with multiple inputs (그림 캡쳐가 안됨 ㅎ)

그래서 텐서가 뭔데.
