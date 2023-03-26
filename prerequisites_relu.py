#%%
import math
from einops import rearrange, repeat, reduce
import torch as t

#%%

def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


def test_relu(relu_func):
    print(f"Testing: {relu_func.__name__}")
    x = t.arange(-1, 3, dtype=t.float32, requires_grad=True)
    out = relu_func(x)
    expected = t.tensor([0.0, 0.0, 1.0, 2.0])
    assert_all_close(out, expected)

#%%
def relu_clone_setitem(x: t.Tensor) -> t.Tensor:
    """Make a copy with torch.clone and then assign to parts of the copy."""
    copy = t.clone(x)
    copy[copy <= 0.0] = 0
    return copy

test_relu(relu_clone_setitem)

#%%

def relu_where(x: t.Tensor) -> t.Tensor:
    """Use torch.where."""
    return t.where(x > 0, x.double(), 0.0).float()


test_relu(relu_where)
#%%

def relu_maximum(x: t.Tensor) -> t.Tensor:
    """Use torch.maximum."""
    return t.maximum(x, t.zeros_like(x))


test_relu(relu_maximum)


def relu_abs(x: t.Tensor) -> t.Tensor:
    """Use torch.abs."""
    return (t.abs(x) + x) / 2


test_relu(relu_abs)


def relu_multiply_bool(x: t.Tensor) -> t.Tensor:
    """Create a boolean tensor and multiply the input by it elementwise."""
    return (x > 0)*x


test_relu(relu_multiply_bool)