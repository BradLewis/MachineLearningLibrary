from machineLearningLibrary.activationFunctions import (Sigmoid,
                                                        Tanh,
                                                        Relu,
                                                        LeakyRelu)


def test_sigmoid():
    sigmoid = Sigmoid()
    assert sigmoid.get(0) == 0.5


def test_tanh():
    tanh = Tanh()
    assert tanh.get(0) == 0


def test_relu():
    relu = Relu()
    assert relu.get(5) == 5
    assert relu.get(-5) == 0


def test_leakyRelu():
    leakyRelu = LeakyRelu(0.1)
    assert leakyRelu.get(5) == 5
    assert leakyRelu.get(-5) == -0.5
