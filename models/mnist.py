import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import scipy.io as sio


class LinearTranspose(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearTranspose, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight.transpose(0, 1), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        self.x = x
        self.conv1_out = self.conv1(self.x)
        self.pool1_out, self.pool1_ind = F.max_pool2d(self.conv1_out, (2, 2), return_indices=True)
        self.conv2_out = self.conv2(self.pool1_out)
        self.pool2_out, self.pool2_ind = F.max_pool2d(self.conv2_out, (2, 2), return_indices=True)
        self.flat_out = self.pool2_out.view(-1, self.num_flat_features(self.pool2_out))
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)

        return self.fc2_out

    def load_weights(self, source=None):
        if source is None:
            source = 'data/mnist-lenet-92dd205e.mat'
        mcn = sio.loadmat(source)
        mcn_weights = dict()

        mcn_weights['conv1.weights'] = mcn['net'][0][0][0][0][0][0][0][1][0][0].transpose()
        mcn_weights['conv1.bias'] = mcn['net'][0][0][0][0][0][0][0][1][0][1].flatten()

        mcn_weights['conv2.weights'] = mcn['net'][0][0][0][0][2][0][0][1][0][0].transpose()
        mcn_weights['conv2.bias'] = mcn['net'][0][0][0][0][2][0][0][1][0][1].flatten()

        mcn_weights['fc1.weights'] = mcn['net'][0][0][0][0][4][0][0][1][0][0].transpose().reshape(500, -1)
        mcn_weights['fc1.bias'] = mcn['net'][0][0][0][0][4][0][0][1][0][1].flatten()

        mcn_weights['fc2.weights'] = mcn['net'][0][0][0][0][6][0][0][1][0][0].transpose().reshape(10, -1)
        mcn_weights['fc2.bias'] = mcn['net'][0][0][0][0][6][0][0][1][0][1].flatten()

        for k in ['conv1', 'conv2', 'fc1', 'fc2']:
            t = self.__getattr__(k)
            assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
            t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k]).cuda()
            assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
            t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k]).cuda()

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class InverseLeNet(nn.Module):
    def __init__(self):
        super(InverseLeNet, self).__init__()
        self.transposefc2 = LinearTranspose(10, 500, bias=False)
        self.transposefc1 = LinearTranspose(500, 50 * 4 * 4, bias=False)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.transposeconv2 = nn.ConvTranspose2d(50, 20, 5, bias=False)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.transposeconv1 = nn.ConvTranspose2d(20, 1, 5, bias=False)

    def forward(self, x, relu1_mask, pool2_ind, pool1_ind):
        self.relu1_out = self.transposefc2(x)
        self.fc1_out = self.relu1_out * relu1_mask
        self.flat_out = self.transposefc1(self.fc1_out)
        self.pool2_out = self.flat_out.view(-1, 50, 4, 4)
        self.conv2_out = self.unpool2(self.pool2_out, pool2_ind)
        self.pool1_out = self.transposeconv2(self.conv2_out)
        self.conv1_out = self.unpool2(self.pool1_out, pool1_ind)
        self.input_out = self.transposeconv1(self.conv1_out)
        return self.input_out

    def copy_from(self, net):
        assert self.transposefc2.weight.data.size() == net.fc2.weight.data.size()
        self.transposefc2.weight = net.fc2.weight

        assert self.transposefc1.weight.data.size() == net.fc1.weight.data.size()
        self.transposefc1.weight = net.fc1.weight

        assert self.transposeconv2.weight.data.size() == net.conv2.weight.data.size()
        self.transposeconv2.weight = net.conv2.weight

        assert self.transposeconv1.weight.data.size() == net.conv1.weight.data.size()
        self.transposeconv1.weight = net.conv1.weight

    def forward_from_net(self, net, input_image, idx):
        num_target_label = idx.size()[1]
        batch_size = input_image.size()[0]
        image_shape = input_image.size()[1:]

        # use inversenet to calculate gradient
        output_var = net(input_image.cuda())

        dzdy = np.zeros((idx.numel(), output_var.size()[1]), dtype=np.float32)
        dzdy[np.arange(idx.numel()), idx.view(idx.numel()).cpu().numpy()] = 1.

        inverse_input_var = torch.from_numpy(dzdy).cuda()
        inverse_input_var.requires_grad = True
        inverse_output_var = self.forward(
            inverse_input_var,
            (net.fc1_out > 0).float().repeat(1, num_target_label).view(idx.numel(), 500),
            net.pool2_ind.repeat(1, num_target_label, 1, 1).view(idx.numel(), 50, 4, 4),
            net.pool1_ind.repeat(1, num_target_label, 1, 1).view(idx.numel(), 20, 12, 12))

        dzdx = inverse_output_var.view(input_image.size()[0], idx.size()[1], -1).transpose(1, 2)
        return dzdx


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        self.x = x
        self.flat_out = self.x.view(-1, 784)
        self.fc1_out = self.fc1(self.flat_out)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out)
        return self.fc3_out

    def load_weights(self, source=None):
        if source is None:
            source = 'data/mnist-mlp-d072f4c8.mat'
        mcn = sio.loadmat(source)
        mcn_weights = dict()

        mcn_weights['fc1.weights'] = mcn['net'][0][0][0][0][0][0][0][1][0][0].transpose().reshape(500, -1)
        mcn_weights['fc1.bias'] = mcn['net'][0][0][0][0][0][0][0][1][0][1].flatten()

        mcn_weights['fc2.weights'] = mcn['net'][0][0][0][0][2][0][0][1][0][0].transpose().reshape(150, -1)
        mcn_weights['fc2.bias'] = mcn['net'][0][0][0][0][2][0][0][1][0][1].flatten()

        mcn_weights['fc3.weights'] = mcn['net'][0][0][0][0][4][0][0][1][0][0].transpose().reshape(10, -1)
        mcn_weights['fc3.bias'] = mcn['net'][0][0][0][0][4][0][0][1][0][1].flatten()

        for k in ['fc1', 'fc2', 'fc3']:
            t = self.__getattr__(k)
            assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
            t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k]).cuda()
            assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
            t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k]).cuda()


class InverseMLP(nn.Module):
    def __init__(self):
        super(InverseMLP, self).__init__()
        self.transposefc3 = LinearTranspose(10, 150, bias=False)
        self.transposefc2 = LinearTranspose(150, 500, bias=False)
        self.transposefc1 = LinearTranspose(500, 784, bias=False)

    def forward(self, x, relu1_mask, relu2_mask):
        self.relu2_out = self.transposefc3(x)
        self.fc2_out = self.relu2_out * relu2_mask
        self.relu1_out = self.transposefc2(self.fc2_out)
        self.fc1_out = self.relu1_out * relu1_mask
        self.flat_out = self.transposefc1(self.fc1_out)
        self.input_out = self.flat_out.view(-1, 1, 28, 28)
        return self.input_out

    def copy_from(self, net):
        for k in ['fc1', 'fc2', 'fc3']:
            t = net.__getattr__(k)
            tt = self.__getattr__('transpose%s' % k)
            assert t.weight.data.size() == tt.weight.data.size()
            tt.weight = t.weight

    def forward_from_net(self, net, input_image, idx):
        num_target_label = idx.size()[1]
        batch_size = input_image.size()[0]
        image_shape = input_image.size()[1:]

        output_var = net(input_image.cuda())

        dzdy = np.zeros((idx.numel(), output_var.size()[1]), dtype=np.float32)
        dzdy[np.arange(idx.numel()), idx.view(idx.numel()).cpu().numpy()] = 1.

        inverse_input_var = torch.from_numpy(dzdy).cuda()
        inverse_input_var.requires_grad = True
        inverse_output_var = self.forward(
            inverse_input_var,
            (net.fc1_out > 0).float().repeat(1, num_target_label).view(idx.numel(), 500),
            (net.fc2_out > 0).float().repeat(1, num_target_label).view(idx.numel(), 150),
        )

        dzdx = inverse_output_var.view(input_image.size()[0], idx.size()[1], -1).transpose(1, 2)
        return dzdx
