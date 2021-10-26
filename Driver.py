import cancer_data
from vec import Vec
import vec
from mat import Mat
import mat
import random
from mat import transpose


def signum(u):
    for n2 in u.D:
        if u[n2] < 0:
            u[n2] = -1
        else:
            u[n2] = 1
    return u


def fraction_wrong(A, b, w):
    length = len(b.D)
    num = b * signum(A*w)
    return ((length-num)/2)/length


def loss(A, b, w):
    return (A*w - b) * (A*w - b)


def find_grad(A, b, w):
    # grad = Vec(A.D[1], {})
    # for n1 in A.D[0]:
    #     temp = Vec(A.D[1], {})
    #     for n2 in A.D[1]:
    #         temp[n2] = A[(n1, n2)]
    #     grad = grad + 2 * (temp * w - b[n1]) * temp
    # return grad
    return 2 * (transpose(A) * ((A * w) - b))


def gradient_decent_step(A, b, w, sigma):
    return w - (sigma * find_grad(A, b, w))


def gradient_decent(A, b, w, sigma, T):
    for i2 in range(T):
        if i2 % 30 == 0:
            print("Loss: " + str(loss(A, b, w)))
            print("Fraction Wrong: " + str(fraction_wrong(A, b, w)))
        w = gradient_decent_step(A, b, w, sigma)
    return w


data = cancer_data.read_training_data('train.data')
A2 = data[0]
b2 = data[1]

w2 = Vec(A2.D[1], {})

for n in A2.D[1]:
    w2[n] = 0

#print(gradient_decent(A2, b2, w2, .000000002, 1000))

for i in range(800):
    if i % 30 == 0:
        print("Loss: " + str(loss(A2, b2, w2)))
        print("Fraction Wrong: " + str(fraction_wrong(A2, b2, w2)))
        #print(w2)
    w2 = gradient_decent_step(A2, b2, w2, .000000001)
    #print(gradient_decent_step(A2, b2, w2, .000000002))
    #print("Fraction Wrong: " + str(fraction_wrong(A2, b2, w2)))
    #print(w3)
    #w2 = w3
    #print(w3)

print(w2)

new_data = cancer_data.read_training_data('validate.data')
A_new = new_data[0]
b_new = new_data[1]

print(fraction_wrong(A_new, b_new, w2))
