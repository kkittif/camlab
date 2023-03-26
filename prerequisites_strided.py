from collections import namedtuple
import math
from einops import rearrange, repeat, reduce
import torch as t


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")

def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")

TestCase = namedtuple("TestCase", ["output", "size", "stride"])
test_input_a = t.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
test_cases = [
    TestCase(output=t.tensor([0, 1, 2, 3]), size=(1,), stride=(1,)),
    TestCase(output=t.tensor([[0, 1, 2], [5, 6, 7]]), size=(1,), stride=(1,)),
    TestCase(output=t.tensor([[0, 0, 0], [11, 11, 11]]), size=(1,), stride=(1,)),
    TestCase(output=t.tensor([0, 6, 12, 18]), size=(1,), stride=(1,)),
    TestCase(output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), size=(1,), stride=(1,)),
    TestCase(
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(1,),
        stride=(1,),
    ),
]
for (i, case) in enumerate(test_cases):
    actual = test_input_a.as_strided(size=case.size, stride=case.stride)
    if (case.output != actual).any():
        print(f"Test {i} failed:")
        print(f"Expected: {case.output}")
        print(f"Actual: {actual}")
    else:
        print(f"Test {i} passed!")