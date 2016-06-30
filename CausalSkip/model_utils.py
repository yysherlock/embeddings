import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    """
    x = x.T - np.max(x.T, axis=0)
    x = np.exp(x) / np.sum(np.exp(x),axis=0)

    return x.T

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    x = 1.0 / (1 + np.exp(-x))
    return x

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x.
    """
    return f * (1-f)

def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        print("---per check--")
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later

        x[ix] += h
        random.setstate(rndstate)
        fx1, grad1 = f(x)
        x[ix] -= 2*h
        random.setstate(rndstate)
        fx2, grad2 = f(x)
        numgrad = (fx1 - fx2) / (2*h)
        x[ix] += h

        print("cost:",fx, 'cost(w-h):',fx1, 'cost(w+h):', fx2)
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return
        else: print("Pass,","Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
        it.iternext() # Step to next dimension

    print("Gradient check passed!")
