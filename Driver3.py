from mat import *
from vec import *
from cancer_data import *


def signum(u):
    return Vec(u.D, {x: 1 if u[x] >= 0 else -1 for x in u.D})


def fraction_wrong(A, b, w):
    v = signum(A*w)
    return sum([1 for x in b.D if b[x] != v[x]]) / len(b.D)


def loss(A, b, w):
    diff_vec = (A * w) - b
    return sum(map(lambda x: x**2, [diff_vec[y] for y in b.D]))


def find_grad(A, b, w):
    return 2 * ((A * w) - b) * A


def gradient_descent_step(A, b, w, sigma):
    return w - (sigma * find_grad(A, b, w))


def gradient_descent(A, b, w, sigma, T):
    for x in range(T):
        if x % 1 == 0:
            print("Loss function: ", loss(A, b, w))
            print("Fraction wrong for w:", fraction_wrong(A, b, w))
        w = gradient_descent_step(A, b, w, sigma)
    return w


data = read_training_data('train.data')
A2 = data[0]
b2 = data[1]

w2 = Vec(A2.D[1], {})

for n in A2.D[1]:
    w2[n] = 1

print(gradient_descent(A2, b2, w2, .0000002, 1000))
