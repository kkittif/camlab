def test_relu(relu_func):
    print(f"Testing: {relu_func.__name__}")
    x = t.arange(-1, 3, dtype=t.float32, requires_grad=True)
    out = relu_func(x)
    expected = t.tensor([0.0, 0.0, 1.0, 2.0])
    assert_all_close(out, expected)


def relu_clone_setitem(x: t.Tensor) -> t.Tensor:
    """Make a copy with torch.clone and then assign to parts of the copy."""
    pass


test_relu(relu_clone_setitem)


def relu_where(x: t.Tensor) -> t.Tensor:
    """Use torch.where."""
    pass


test_relu(relu_where)


def relu_maximum(x: t.Tensor) -> t.Tensor:
    """Use torch.maximum."""
    pass


test_relu(relu_maximum)


def relu_abs(x: t.Tensor) -> t.Tensor:
    """Use torch.abs."""
    pass


test_relu(relu_abs)


def relu_multiply_bool(x: t.Tensor) -> t.Tensor:
    """Create a boolean tensor and multiply the input by it elementwise."""
    pass


test_relu(relu_multiply_bool)