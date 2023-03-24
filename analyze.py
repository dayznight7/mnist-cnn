import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


with open("train_loss_history.pkl", 'rb') as f:
    train_loss_history = pickle.load(f)

x = list(range(1, len(train_loss_history)+1))
plt.plot(x, train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # liner 가중치 xavier 로 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out.size(0) = batch size
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def imgshow(img):
    np_img = img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    plt.imshow(np_img)
    plt.show()


batch_size = 4
mnist_train = dsets.MNIST(root="MNIST_data/",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

data_loader2 = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=100,
                                          shuffle=False,
                                          drop_last=True)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = CNN()
net.to(device)
net.load_state_dict(torch.load("mnist_cnn.pth"))


total = 0
wrong_img = []
wrong_label = []
wrong_predict = []

with torch.no_grad():
    for images, labels in data_loader2:
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        _, predicts = torch.max(output, 1)
        for image, label, predict in zip(images, labels, predicts):
            if label != predict:
                wrong_img.append(image.cpu().numpy())
                wrong_label.append(label.cpu().numpy())
                wrong_predict.append(predict.cpu().numpy())
            total += 1

print(total)
print(len(wrong_label))

with open("wrong_img_label_predict.pkl", "wb") as f:
    pickle.dump((wrong_img, wrong_label, wrong_predict), f)

