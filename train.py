import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100


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


model = CNN().to(device)
# softmax already in Loss function
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))


train_loss_history = []

for epoch in range(training_epochs):
    avg_cost = 0

    # shuffle on 되있는 data_loader, 매 에폭마다 배치 군집이 달라짐
    for X, Y in data_loader:
        # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        # softmax 함수 criterion 안에 들어있음
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    train_loss_history.append(avg_cost.item())
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    # test_label 을 target 으로 바꾸라는데 test_label 이 더 맞는 표현 아닌가

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())


# 학습된 가중치 저장

PATH = './mnist_cnn.pth'
torch.save(model.state_dict(), PATH)

# train_loss_history 저장

import pickle

with open("train_loss_history.pkl", "wb") as f:
    pickle.dump(train_loss_history, f)

