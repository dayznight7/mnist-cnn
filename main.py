import torch
import torch.nn as nn


inputs = torch.Tensor(1, 1, 28, 28)
print("input : " + str(inputs.shape))

conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)


conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)


pool = nn.MaxPool2d(2)
print(pool)


out = conv1(inputs)
print("input * conv1 : " + str(out.shape))


out = pool(out)
print("input * conv1 * pool : " + str(out.shape))


out = conv2(out)
print("input * conv1 pool * conv2 : " + str(out.shape))


out = pool(out)
print("input * conv1 pool * conv2 pool : " + str(out.shape))


out = out.view(out.size(0), -1)
print(out.shape)


fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)


# 참고문헌
# https://wikidocs.net/63565

