from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import math
from torch.autograd import Variable

groups_x = 5
groups_y = 5


# https://stackoverflow.com/questions/61629395/how-to-prune-weights-less-than-a-threshold-in-pytorch
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    #def __init__(self, threshold):
    #    self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > 0.001 #self.threshold


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)  # Randomly zeroes elements of input tensor with probability P
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def split_in_groups(weights, groups):
    for i in range(0, groups_x):
        for j in range(0, groups_y):
            x_start = math.floor(weights.size(0) / groups_x * i)
            x_end = math.floor(weights.size(0) / groups_x * (i + 1))
            y_start = math.floor(weights.size(1) / groups_y * j)
            y_end = math.floor(weights.size(1) / groups_y * (j + 1))

            if i != 0:
                x_start += 1

            if j != 0:
                y_start += 1

            if i == groups_x - 1:
                x_end = weights.size(0) - 1

            if j == groups_y - 1:
                y_end = weights.size(1) - 1

            groups[i][j] = weights[x_start:x_end, y_start:y_end]


def compute_group_lasso(groups, lambda_g):
    loss = torch.tensor(0, dtype=torch.float32, requires_grad=True, device="cuda")

    i = 0

    for x in range(0, groups_x):
        for y in range(0, groups_y):
            if i % 2 == 0:  # Only include half of the groups in the loss function
                loss = torch.add(loss, torch.mul(torch.sqrt(torch.sum(torch.square(groups[x][y]))),
                                             torch.tensor(lambda_g, device="cuda")))
            i = i + 1

    return loss


def threshold_pruning(module, name):
    ThresholdPruning.apply(module, name)
    #ThresholdPruning(0.1).apply(module, name)
    return module


def plot(model):
    weights_flattened = model.fc1.weight.data.cpu().numpy().flatten()

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    n, bins, patches = plt.hist(weights_flattened, bins=300)
    plt.title("Distribution of all weights of fc1")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of occurences")
    ax = plt.gca()
    ax.set_ylim([0, 100000])
    xlim = ax.get_xlim()
    plt.show()

    weights = model.fc1.weight.data

    groups = np.empty((groups_x, groups_y), dtype=object)

    split_in_groups(weights, groups)

    fig, axs = plt.subplots(groups_x, groups_y)
    fig.subplots_adjust(wspace=1, hspace=1)

    ylim = [0, 0]

    for x in range(0, groups_x):
        for y in range(0, groups_y):
            axs[x][y].hist(groups[x][y].cpu().numpy().flatten(), bins=100)
            axs[x][y].tick_params(axis='both', labelsize=7)
            ylim[1] = max(ylim[1], axs[x][y].get_ylim()[1])
            axs[y][x].set_xlabel("Weight", fontsize=4)
            axs[y][x].set_ylabel("Number of occurrences", fontsize=4)

    fig.suptitle("Distribution of weights over the "+str(groups_x)+"x"+str(groups_y)+" groups")
    plt.setp(axs, xlim=xlim, ylim=[0, 1000])

    plt.show()

    plt.imshow(weights.cpu().numpy(), cmap=plt.get_cmap('Blues'), aspect='auto', interpolation='sinc')
    plt.xlabel('In features')
    plt.ylabel('Out features')
    plt.title('Heatmap of weights of fc1, linear transformation')
    plt.colorbar()
    plt.show()

    print("Number of nonzero weights: ")
    print(torch.count_nonzero(weights))

    return

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    #group_lasso_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True, device="cuda")
    #print(group_lasso_loss)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        weights_conv1 = model.conv1.weight.data
        weights_conv2 = model.conv2.weight.data
        weights_fc1 = model.fc1.weight.data
        weights_fc2 = model.fc2.weight.data

        threshold_pruning(model.fc1, "weight")

        #thresholdPos = nn.Threshold(0.001, 0, inplace=True)
        #thresholdNeg = nn.Threshold(-0.001, 0, inplace=True)
        #thresholdPos(weights_fc1)
        #thresholdNeg(weights_fc2)

        #for x in range(0, weights_fc1.size(0)):
        #    for y in range(0, weights_fc1.size(1)):
        #        if 0.001 > weights_fc1[x][y] > -0.001:
        #            weights_fc1[x][y] = torch.tensor(0, dtype=torch.float32, device="cuda")

        fc1_groups = np.empty((groups_x, groups_y), dtype=object)
        fc2_groups = np.empty((groups_x, groups_y), dtype=object)

        split_in_groups(weights_fc1, fc1_groups)
        # split_in_groups(weights_fc2, fc2_groups)

        group_lasso_loss = compute_group_lasso(fc1_groups, 1)
        regular_loss = F.nll_loss(output, target)

        #print(regular_loss)

        #print(group_lasso_loss)

        loss = torch.add(regular_loss, group_lasso_loss)
        #print(loss)

        # loss = F.nll_loss(output, target)  # Old loss function

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def atest(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)  #TODO change to SGD
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)

        # threshold_pruning(model.fc1, "weight")

        atest(model, device, test_loader)
        scheduler.step()

    # Create plot
    plot(model)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

    net = Net()

    params = list(net.parameters())
