# -*- coding:utf-8 -*-

from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from DataLoader import get_training_set, get_test_set
import torchvision


# Training settings
class option:

    def __init__(self):
        self.cuda = True  # use cuda?
        # change True.
        self.batchSize = 1  # training batch size
        self.testBatchSize = 1  # testing batch size
        self.nEpochs = 100  # umber of epochs to train for
        self.lr = 0.001  # Learning Rate. Default=0.01
        self.threads = 8  # number of threads for data loader to use
        self.seed = 123  # random seed to use. Default=123
        self.size = 128 # change, 128.
        # self.remsize = 20 # change, temporarily not need
        self.colordim = 1
        self.save_step = 40 # In training, save the training results
        self.target_mode = 'seg' # seg task.
        self.pretrain_net = "/root/datasets/CNN/model_epoch_1.pth"
        # change pretrain_net. default pretrained is False.


def map01(tensor, eps=1e-5):
    # input/output:tensor
    max = np.max(tensor.numpy(), axis=(1, 2, 3), keepdims=True)
    min = np.min(tensor.numpy(), axis=(1, 2, 3), keepdims=True)
    if (max - min).any():
        return torch.from_numpy((tensor.numpy() - min) / (max - min + eps))
    else:
        return torch.from_numpy((tensor.numpy() - min) / (max - min))


def sizeIsValid(size):
    for i in range(4):
        size -= 4
        if size % 2:
            return 0
        else:
            size /= 2
    for i in range(4):
        size -= 4
        size *= 2
    return size - 4


opt = option()

# change! This code do what???
# May be, it want to random crop!!
# 这的代码也许会用到, 但是需要修改!
'''
target_size = sizeIsValid(opt.size)
print("outputsize is: " + str(target_size))
if not target_size:
    raise Exception("input size invalid")
target_gap = (opt.size - target_size) // 2
'''


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(
    opt.size, target_mode=opt.target_mode, colordim=opt.colordim)
test_set = get_test_set(
    opt.size, target_mode=opt.target_mode, colordim=opt.colordim)
training_data_loader = DataLoader(
    dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
# TODO: change the testing_data_loader. 
# atfer having the testing datasts and 
# testing_data_loader = DataLoader(
#     dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building unet')
unet = UNet(opt.colordim)


criterion = nn.SoftMarginLoss()
if cuda:
    unet = unet.cuda()
    criterion = criterion.cuda()

pretrained = False
# change False.
if pretrained:
    unet.load_state_dict(torch.load(opt.pretrain_net))

optimizer = optim.Adam(unet.parameters(), lr=opt.lr)
print('===> Training unet')


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader):
        # randH = random.randint(0, opt.remsize)
        # randW = random.randint(0, opt.remsize)
        # change! use batch directly!
        '''
        # This code might used, but it must modify!!
        input = Variable(batch[0][:, :, randH:randH +
                                  opt.size, randW:randW + opt.size])
        target = Variable(batch[1][:, :,
                                   randH + target_gap:randH + target_gap + target_size,
                                   randW + target_gap:randW + target_gap + target_size])
        '''
        input_batch = Variable(batch[0])
        # print(input_batch.shape)
        target = Variable(batch[1])
        # print(target.shape)

        # target =target.squeeze(1)
        # print(target.data.size())
        if cuda:
            input_batch = input_batch.cuda()
            target = target.cuda()
        output = unet.forward(input_batch)
        # print(input_batch.data.size())
        loss = criterion(output, target)
        epoch_loss += loss.data[0]
        # change! train process
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 40 is 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch,
                                                               iteration, len(training_data_loader), loss.data[0]))
        # save training results
        if iteration % opt.save_step == 0:
            imgout = output.data / 2 + 1
            # change! can use os.path.exists() to distinguish a floder/file exit or not.
            torchvision.utils.save_image(
                imgout, "/root/result/epch_" + str(epoch) + "_" + str(iteration) + '.jpg')
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def test():
    totalloss = 0
    for batch in testing_data_loader:
        input = Variable(batch[0], volatile=True)
        target = Variable(batch[1][:, :,
                                   target_gap:target_gap + target_size,
                                   target_gap:target_gap + target_size],
                          volatile=True)
        #target =target.long().squeeze(1)
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = unet(input)
        loss = criterion(prediction, target)
        totalloss += loss.data[0]
    print("===> Avg. test loss: {:.4f} dB".format(
        totalloss / len(testing_data_loader)))


def checkpoint(epoch):
    # change the save checkpoint path.
    # change! can use os.path.exists() to distinguish a floder/file exit or not.
    model_out_path = "/root/model/model_epoch_{}.pth".format(
        epoch)
    torch.save(unet.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# change, add main function.
if __name__ == '__main__':
    for epoch in range(1, opt.nEpochs + 1):
        # change, if you use pretrained model, it can be modify the epoch! 
        train(epoch)
        if epoch % 1 == 0:
            checkpoint(epoch)
        # TODO: atfer change the BSDDataLoader.py get_test_set() function, you can use test().
        # test()
