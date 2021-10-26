from neural_network import NeuralNetwork
import cancer_data
import matutil
from mat import *
import sys
import vec

import torch

data = cancer_data.read_training_data('all_data.data')
A = data[0]
b = data[1]

data2 = cancer_data.read_training_data('validate.data')
A2 = data2[0]
b2 = data2[1]


network = NeuralNetwork(no_of_in_nodes=30,
                               no_of_out_nodes=2,
                               no_of_hidden_nodes=16,
                               learning_rate=0.1,
                               bias=None)

A_vec = matutil.mat2rowdict(A)
A_vec_features = matutil.mat2coldict(A)
print(A_vec)
params = ["radius", "texture", "perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]
stats = ["(mean)", "(stderr)", "(worst)"]
domain = [y+x for y in params for x in stats]

feature_stats = Mat((set(domain), {'range', 'mean'}), {})
for n in domain:
    cur_data = A_vec_features[n]
    sum2 = 0
    max2 = -sys.maxsize
    min2 = sys.maxsize
    for n2 in cur_data.D:
        sum2 += cur_data[n2]
        max2 = max(max2, cur_data[n2])
        min2 = min(min2, cur_data[n2])
    feature_stats[n, 'range'] = max2-min2
    feature_stats[n, 'mean'] = sum2 / (len(cur_data.D))


count1 = 0
count2 = 0
features = [[0 for z in range(30)] for k in range(len(A.D[0]))]
end_vec = [[0, 0] for k in range(len(A.D[0]))]  # [1, 0] if yes, [0, 1] if no
for n in A_vec:
    count2 = 0
    for n2 in domain:
        features[count1][count2] = (A_vec[n][n2]-feature_stats[n2, 'mean']) / feature_stats[n2, 'range']
        count2 = count2 + 1
    end_vec[count1][0] = max(0, b[n])
    end_vec[count1][1] = max(0, (-b[n]))
    count1 = count1 + 1

# print(features[0])
# print(end_vec[0])
# network.train(features[0], end_vec[0])
# network.train(features[1], end_vec[1])

for l in range(200):
    for i in range(299):
        network.train(features[i], end_vec[i])


end_list = [0 for k in range(259)]
dictionary = {}

for i in range(300, 559):
    result = network.run(features[i])
    # print(str(result))
    end_list[i-300] = result[0][0]
    dictionary[end_list[i-300]] = end_vec[i][0]
    # if end_vec[i][0] == 1:
    #     print('YES')
    # else:
    #     print("NO")
    # print("")

end_list.sort()
for n in end_list:
    print(n)
    ans = dictionary[n]
    print(ans)
    if ans == 1:
        print("THERE IS A 1 HERE, THERE IS A 1 HERE") #about 18 incorrectly placed  8/259 = .03 wrong
    print("")

#https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/neural_networks_tutorial.py
