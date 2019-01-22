import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io as sio
import glog as log


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)
        self.conv5 = nn.Conv2d(64, 10, 1)

        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal(w.weight.data)
            w.bias.data.fill_(0)

        self.out = dict()

    def save(self, x, name):
        self.out[name] = x

    def forward(self, x):
        self.save(x, 'x')
        x = self.conv1(x)
        self.save(x, 'conv1_out')
        x, pool1_ind = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        self.save(x, 'pool1_out')
        self.save(pool1_ind, 'pool1_ind')
        x = F.relu(x)
        self.save(x, 'relu1_out')

        x = self.conv2(x)
        self.save(x, 'conv2_out')
        x = F.relu(x)
        self.save(x, 'relu2_out')
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        self.save(x, 'pool2_out')

        x = self.conv3(x)
        self.save(x, 'conv3_out')

        x = F.relu(x)
        self.save(x, 'relu3_out')
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        self.save(x, 'pool3_out')

        x = self.conv4(x)
        self.save(x, 'conv4_out')
        x = F.relu(x)
        self.save(x, 'relu4_out')

        x = self.conv5(x)
        self.save(x, 'conv5_out')

        x = x.view(-1, 10)
        self.save(x, 'flat_out')

        return x

    def load_weights(self, source=None):
        if source is None:
            source = 'data/cifar10-convnet-15742544.mat'
        if source.endswith('mat'):
            log.info('Load cifar10 weights from matlab model %s' % source)
            mcn = sio.loadmat(source)
            mcn_weights = dict()

            mcn_weights['conv1.weights'] = mcn['net'][0][0][0][0][0][0][0][1][0][0].transpose()
            mcn_weights['conv1.bias'] = mcn['net'][0][0][0][0][0][0][0][1][0][1].flatten()

            mcn_weights['conv2.weights'] = mcn['net'][0][0][0][0][3][0][0][1][0][0].transpose()
            mcn_weights['conv2.bias'] = mcn['net'][0][0][0][0][3][0][0][1][0][1].flatten()

            mcn_weights['conv3.weights'] = mcn['net'][0][0][0][0][6][0][0][1][0][0].transpose()
            mcn_weights['conv3.bias'] = mcn['net'][0][0][0][0][6][0][0][1][0][1].flatten()

            mcn_weights['conv4.weights'] = mcn['net'][0][0][0][0][9][0][0][1][0][0].transpose()
            mcn_weights['conv4.bias'] = mcn['net'][0][0][0][0][9][0][0][1][0][1].flatten()

            mcn_weights['conv5.weights'] = mcn['net'][0][0][0][0][11][0][0][1][0][0].transpose()
            mcn_weights['conv5.bias'] = mcn['net'][0][0][0][0][11][0][0][1][0][1].flatten()

            for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                t = self.__getattr__(k)
                assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
                t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k])
                assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
                t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k])
        elif source.endswith('pth'):
            log.info('Load cifar10 weights from PyTorch model %s' % source)
            pth_weights = torch.load(source)
            for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                t = self.__getattr__(k)
                assert t.weight.data.size() == pth_weights['net.%s.weight' % k].shape
                t.weight.data[:] = pth_weights['net.%s.weight' % k]
                assert t.bias.data.size() == pth_weights['net.%s.bias' % k].shape
                t.bias.data[:] = pth_weights['net.%s.bias' % k]


class InverseConvNet(nn.Module):
    def __init__(self):
        super(InverseConvNet, self).__init__()
        self.transposeconv5 = nn.ConvTranspose2d(10, 64, 1, bias=False)
        self.transposeconv4 = nn.ConvTranspose2d(64, 64, 4, bias=False)
        self.transposeconv3 = nn.ConvTranspose2d(64, 32, 5, padding=2, bias=False)
        self.transposeconv2 = nn.ConvTranspose2d(32, 32, 5, padding=2, bias=False)
        self.transposeconv1 = nn.ConvTranspose2d(32, 3, 5, padding=2, bias=False)

        # inverse pool2 (average pooling)
        self.w2 = torch.zeros(32, 32, 2, 2)
        for i in range(32):
            self.w2[i, i, :, :] = 0.25

        # inverse pool3 (average pooling)
        self.w3 = torch.zeros(64, 64, 2, 2)
        for i in range(64):
            self.w3[i, i, :, :] = 0.25

        self.out = dict()

    def save(self, x, name):
        self.out[name] = x

    def forward(self, x, pool1_ind, relu1_mask, relu2_mask, relu3_mask, relu4_mask):
        x = x.view(-1, 10, 1, 1)
        self.save(x, 'conv5_out')

        x = self.transposeconv5(x)
        self.save(x, 'relu4_out')
        x = x * relu4_mask
        self.save(x, 'conv4_out')

        x = self.transposeconv4(x)
        self.save(x, 'pool3_out')
        x = F.conv_transpose2d(x, self.w3, stride=2)
        self.save(x, 'relu3_out')
        x = x * relu3_mask
        self.save(x, 'conv3_out')

        x = self.transposeconv3(x)
        self.save(x, 'pool2_out')
        x = F.conv_transpose2d(x, self.w2, stride=2)
        self.save(x, 'relu2_out')
        x = x* relu2_mask
        self.save(x, 'conv2_out')

        x = self.transposeconv2(x)
        self.save(x, 'relu1_out')
        x = x * relu1_mask
        self.save(x, 'pool1_out')
        x = F.max_unpool2d(x, pool1_ind, kernel_size=2, stride=2)
        self.save(x, 'conv1_out')
        x = self.transposeconv1(x)
        self.save(x, 'input_out')

        return x

    def copy_from(self, net):
        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            t = net.__getattr__(k)
            tt = self.__getattr__('transpose%s' % k)
            assert t.weight.size() == tt.weight.size()
            tt.weight = t.weight

    def forward_from_net(self, net, input_image, idx):
        num_target_label = idx.size()[1]
        batch_size = input_image.size()[0]
        image_shape = input_image.size()[1:]

        output_var = net(input_image)

        dzdy = np.zeros((idx.numel(), output_var.size()[1]), dtype=np.float32)
        dzdy[np.arange(idx.numel()), idx.view(idx.numel()).cpu().numpy()] = 1.

        inverse_input_var = torch.from_numpy(dzdy).to(input_image.device)
        inverse_input_var.requires_grad = True
        inverse_output_var = self.forward(
            inverse_input_var,
            net.out['pool1_ind'].repeat(1, num_target_label, 1, 1).view(idx.numel(), 32, 16, 16),
            (net.out['pool1_out'] > 0).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 32, 16, 16),
            (net.out['conv2_out'] > 0).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 32, 16, 16),
            (net.out['conv3_out'] > 0).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 64, 8, 8),
            (net.out['conv4_out'] > 0).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 64, 1, 1),
        )

        dzdx = inverse_output_var.view(input_image.size()[0], idx.size()[1], -1).transpose(1, 2)
        return dzdx


class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, 5, padding=2)
        self.cccp1 = nn.Conv2d(192, 160, 1)
        self.cccp2 = nn.Conv2d(160, 96, 1)

        self.conv2 = nn.Conv2d(96, 192, 5, padding=2)
        self.cccp3 = nn.Conv2d(192, 192, 1)
        self.cccp4 = nn.Conv2d(192, 192, 1)

        self.conv3 = nn.Conv2d(192, 192, 3, padding=1)
        self.cccp5 = nn.Conv2d(192, 192, 1)
        self.cccp6 = nn.Conv2d(192, 10, 1)

        for k in ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal(w.weight.data)
            w.bias.data.fill_(0)

        # self.cccp6.weight.data[:] = 0.1 * self.cccp6.weight.data[:]

        self.p = 0.5  # dropout probability
        raise NotImplementedError('Code for NIN will be released soon since we need to '
                                  'clean up our codebase for dropout support')

    def forward(self, x, drop1_mask=None, drop2_mask=None):
        self.x = x
        self.conv1_out = self.conv1(self.x)
        self.relu1_out = F.relu(self.conv1_out)
        self.cccp1_out = self.cccp1(self.relu1_out)
        self.relu_cccp1_out = F.relu(self.cccp1_out)
        self.cccp2_out = self.cccp2(self.relu_cccp1_out)
        self.relu_cccp2_out = F.relu(self.cccp2_out)
        self.pool1_out, self.pool1_ind = F.max_pool2d(self.relu_cccp2_out, kernel_size=2, stride=2, return_indices=True)
        if self.training:
            # when dropout mask passed from outside is None, we need to generate a new dropout mask in this round
            if drop1_mask is None:
                # check if we can re-use previous mask Variable
                # if yes, we simply fill it with bernoulli noise without cloning it, which may save some running time
                if hasattr(self, 'drop1_mask') \
                        and self.drop1_mask is not None \
                        and self.drop1_mask.size() == self.pool1_out.size():
                    drop1_mask = self.drop1_mask
                else:
                    drop1_mask = self.pool1_out.clone().detach()
                drop1_mask.data.bernoulli_(self.p).div_(1. - self.p)
            self.drop1_out = self.pool1_out * drop1_mask
        else:
            self.drop1_out = self.pool1_out
        self.drop1_mask = drop1_mask

        self.conv2_out = self.conv2(self.drop1_out)
        self.relu2_out = F.relu(self.conv2_out)
        self.cccp3_out = self.cccp3(self.relu2_out)
        self.relu_cccp3_out = F.relu(self.cccp3_out)
        self.cccp4_out = self.cccp4(self.relu_cccp3_out)
        self.relu_cccp4_out = F.relu(self.cccp4_out)
        self.pool2_out = F.avg_pool2d(self.relu_cccp4_out, kernel_size=2, stride=2)
        if self.training:
            if drop2_mask is None:
                if hasattr(self, 'drop2_mask') \
                        and self.drop2_mask is not None \
                        and self.drop2_mask.size() == self.pool2_out.size():
                    drop2_mask = self.drop2_mask
                else:
                    drop2_mask = self.pool2_out.clone().detach()
                drop2_mask.data.bernoulli_(self.p).div_(1. - self.p)
            self.drop2_out = self.pool2_out * drop2_mask
        else:
            self.drop2_out = self.pool2_out
        self.drop2_mask = drop2_mask

        self.conv3_out = self.conv3(self.drop2_out)
        self.relu3_out = F.relu(self.conv3_out)
        self.cccp5_out = self.cccp5(self.relu3_out)
        self.relu_cccp5_out = F.relu(self.cccp5_out)
        self.cccp6_out = self.cccp6(self.relu_cccp5_out)
        self.pool3_out = F.avg_pool2d(self.cccp6_out, kernel_size=8)
        self.flat_out = self.pool3_out.view(-1, 10)
        return self.flat_out

    def load_weights(self, source=None):
        if source is None:
            source = 'data/cifar10-nin-62053fa9.mat'
        mcn = sio.loadmat(source)
        mcn_weights = dict()

        mcn_weights['conv1.weights'] = mcn['net'][0][0][0][0][0][0][0][2][0][0].transpose()
        mcn_weights['conv1.bias'] = mcn['net'][0][0][0][0][0][0][0][2][0][1].flatten()

        mcn_weights['cccp1.weights'] = mcn['net'][0][0][0][0][2][0][0][2][0][0].transpose()
        mcn_weights['cccp1.bias'] = mcn['net'][0][0][0][0][2][0][0][2][0][1].flatten()

        mcn_weights['cccp2.weights'] = mcn['net'][0][0][0][0][4][0][0][2][0][0].transpose()
        mcn_weights['cccp2.bias'] = mcn['net'][0][0][0][0][4][0][0][2][0][1].flatten()

        mcn_weights['conv2.weights'] = mcn['net'][0][0][0][0][7][0][0][2][0][0].transpose()
        mcn_weights['conv2.bias'] = mcn['net'][0][0][0][0][7][0][0][2][0][1].flatten()

        mcn_weights['cccp3.weights'] = mcn['net'][0][0][0][0][9][0][0][2][0][0].transpose()
        mcn_weights['cccp3.bias'] = mcn['net'][0][0][0][0][9][0][0][2][0][1].flatten()

        mcn_weights['cccp4.weights'] = mcn['net'][0][0][0][0][11][0][0][2][0][0].transpose()
        mcn_weights['cccp4.bias'] = mcn['net'][0][0][0][0][11][0][0][2][0][1].flatten()

        mcn_weights['conv3.weights'] = mcn['net'][0][0][0][0][14][0][0][2][0][0].transpose()
        mcn_weights['conv3.bias'] = mcn['net'][0][0][0][0][14][0][0][2][0][1].flatten()

        mcn_weights['cccp5.weights'] = mcn['net'][0][0][0][0][16][0][0][2][0][0].transpose()
        mcn_weights['cccp5.bias'] = mcn['net'][0][0][0][0][16][0][0][2][0][1].flatten()

        mcn_weights['cccp6.weights'] = mcn['net'][0][0][0][0][18][0][0][2][0][0].transpose()
        mcn_weights['cccp6.bias'] = mcn['net'][0][0][0][0][18][0][0][2][0][1].flatten()

        for k in ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6']:
            t = self.__getattr__(k)
            assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
            t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k])
            assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
            t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k])


class InverseNIN(nn.Module):
    def __init__(self):
        super(InverseNIN, self).__init__()

        self.transposecccp6 = nn.ConvTranspose2d(10, 192, 1, bias=False)
        self.transposecccp5 = nn.ConvTranspose2d(192, 192, 1, bias=False)
        self.transposeconv3 = nn.ConvTranspose2d(192, 192, 3, padding=1, bias=False)

        self.transposecccp4 = nn.ConvTranspose2d(192, 192, 1, bias=False)
        self.transposecccp3 = nn.ConvTranspose2d(192, 192, 1, bias=False)
        self.transposeconv2 = nn.ConvTranspose2d(192, 96, 5, padding=2, bias=False)

        self.transposecccp2 = nn.ConvTranspose2d(96, 160, 1, bias=False)
        self.transposecccp1 = nn.ConvTranspose2d(160, 192, 1, bias=False)
        self.transposeconv1 = nn.ConvTranspose2d(192, 3, 5, padding=2, bias=False)

        # inverse pool2 (average pooling)
        self.w = torch.zeros(192, 192, 2, 2)
        for i in range(192):
            self.w[i, i, :, :] = 1. / 4

    def forward(self, x, relu1_mask, relu_cccp1_mask, relu_cccp2_mask, pool1_ind, drop1_mask, relu2_mask,
                relu_cccp3_mask, relu_cccp4_mask, drop2_mask, relu3_mask, relu_cccp5_mask):
        batch_size = x.size()[0]
        self.pool3_out = x.view(-1, 10, 1, 1)
        self.cccp6_out = self.pool3_out.expand(batch_size, 10, 8, 8) / 64.
        self.relu_cccp5_out = self.transposecccp6(self.cccp6_out)
        self.cccp5_out = self.relu_cccp5_out * relu_cccp5_mask
        self.relu3_out = self.transposecccp5(self.cccp5_out)
        self.conv3_out = self.relu3_out * relu3_mask
        self.drop2_out = self.transposeconv3(self.conv3_out)
        if self.training:
            self.pool2_out = self.drop2_out * drop2_mask
        else:
            self.pool2_out = self.drop2_out

        self.relu_cccp4_out = F.conv_transpose2d(self.pool2_out, self.w, stride=2)  # inverse pool2 (average pooling)
        self.cccp4_out = self.relu_cccp4_out * relu_cccp4_mask
        self.relu_cccp3_out = self.transposecccp4(self.cccp4_out)
        self.cccp3_out = self.relu_cccp3_out * relu_cccp3_mask
        self.relu2_out = self.transposecccp3(self.cccp3_out)
        self.conv2_out = self.relu2_out * relu2_mask
        self.drop1_out = self.transposeconv2(self.conv2_out)
        if self.training:
            self.pool1_out = self.drop1_out * drop1_mask
        else:
            self.pool1_out = self.drop1_out

        self.relu_cccp2_out = F.max_unpool2d(self.pool1_out, pool1_ind, kernel_size=2, stride=2)
        self.cccp2_out = self.relu_cccp2_out * relu_cccp2_mask
        self.relu_cccp1_out = self.transposecccp2(self.cccp2_out)
        self.cccp1_out = self.relu_cccp1_out * relu_cccp1_mask
        self.relu1_out = self.transposecccp1(self.cccp1_out)
        self.conv1_out = self.relu1_out * relu1_mask
        self.input_out = self.transposeconv1(self.conv1_out)
        return self.input_out

    def copy_from(self, net):
        for k in ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6']:
            t = net.__getattr__(k)
            tt = self.__getattr__('transpose%s' % k)
            assert t.weight.size() == tt.weight.size()
            tt.weight = t.weight

    def forward_from_net(self, net, input_image, idx, drop1_mask=None, drop2_mask=None):
        idx = idx.contiguous()
        num_target_label = idx.size()[1]
        batch_size = input_image.size()[0]
        image_shape = input_image.size()[1:]

        output_var = net(input_image, drop1_mask=drop1_mask, drop2_mask=drop2_mask)

        dzdy = np.zeros((idx.numel(), output_var.size()[1]), dtype=np.float32)
        dzdy[np.arange(idx.numel()), idx.view(idx.numel()).cpu().numpy()] = 1.

        inverse_input_var = torch.from_numpy(dzdy).to(input_image.device)
        inverse_input_var.requires_grad = True
        inverse_output_var = self.forward(
            inverse_input_var,
            (net.conv1_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 32, 32),
            (net.cccp1_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 160, 32, 32),
            (net.cccp2_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 96, 32, 32),
            net.pool1_ind.repeat(1, num_target_label, 1, 1).view(idx.numel(), 96, 16, 16),
            net.drop1_mask.repeat(1, num_target_label, 1, 1).view(idx.numel(), 96, 16, 16)
            if net.drop1_mask is not None else None,
            (net.conv2_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 16, 16),
            (net.cccp3_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 16, 16),
            (net.cccp4_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 16, 16),
            net.drop2_mask.repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 8, 8)
            if net.drop2_mask is not None else None,
            (net.conv3_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 8, 8),
            (net.cccp5_out > 0.).float().repeat(1, num_target_label, 1, 1).view(idx.numel(), 192, 8, 8),
        )

        dzdx = inverse_output_var.view(input_image.size()[0], idx.size()[1], -1).transpose(1, 2)
        return dzdx
