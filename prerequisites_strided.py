from collections import namedtuple

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