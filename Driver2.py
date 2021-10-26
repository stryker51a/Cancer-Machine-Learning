import cancer_data
import vec
import vecutil
import matutil
import math

test_ident = matutil.rowdict2mat({
    0: vecutil.list2vec([1, 0, 0]),
    1: vecutil.list2vec([0, 1, 0]),
    2: vecutil.list2vec([0, 0, 1]),
  })
test_1 = vecutil.list2vec([-1, -1, -1])
test_2 = vecutil.list2vec([1, 1, 1])


def read_training_data():
    return cancer_data.read_training_data('train.data')


def signum(u):
    """
    input: a Vec u
    output: the Vec v with the same domain as u such that
          +1 if u[d] >= 0
    v[d]= {
          -1 if u[d] < 0
    """
    return vec.Vec(u.D, {d: 1 if u[d] >= 0 else -1 for d in u.D})


assert(signum(vec.Vec({'A', 'B'}, {'A': 3, 'B': -2})) == vec.Vec({'A', 'B'}, {'A': 1, 'B': -1}))


def fraction_wrong(A, b, w):
    """
    input: An R x C matrix A whose rows are feature vectors,
    an R-vector b whose entries are +1 and -1,
    and a C-vector w.
    output: The fraction of row labels r of A such that
    the sign of (row r of A) * w differs from that of b[r]
    """
    Aw = A * w
    signs = signum(Aw)
    result = ((signs * b) - len(b.D))/(-2 * len(b.D))
    return abs(result)


assert(fraction_wrong(test_ident, test_1, test_2) == 1)


def loss(A, b, w):
    """
    input: training data A, predictions b, hypothesis vector w
    output: loss calculation on w
    """
    term = (A * w) - b
    return term * term


test_loss = loss(test_ident, test_1, test_2) 
assert(abs(12 - test_loss) < 1e-10)


def find_grad(A, b, w):
    return 2 * (A.transpose() * ((A * w) - b))


def gradient_descent_step(A, b, w, sigma):
    return w - sigma*find_grad(A, b, w)


def gradient_descent(A, b, w, sigma, T):
    for i in range(T):
        w = gradient_descent_step(A, b, w, sigma)
        if (i % 30) == 0:
            print("GRADIENT STATE:\nloss: %s\nfraction wrong: %s\n\n" % (loss(A, b, w), fraction_wrong(A, b, w)))
    return w


A, b = read_training_data()
R = A.D[0]
C = A.D[1]

all_ones = vec.Vec(C, {d: 1 for d in C})
zero_vector = vec.Vec(C, {})
print(gradient_descent(A, b, zero_vector, 1e-9, 900))
