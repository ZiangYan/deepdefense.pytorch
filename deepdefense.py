#!/usr/bin/env python
import sys
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.mnist import LeNet, InverseLeNet, MLP, InverseMLP
from models.cifar10 import ConvNet, InverseConvNet, NIN, InverseNIN


logger = logging.getLogger()


class DeepFool(nn.Module):
    def __init__(self, args):
        super(DeepFool, self).__init__()

        self.device = args.device
        self.num_labels = 10
        self.overshot = 0.02
        self.max_iter = args.max_iter

        # initialize net
        if args.dataset == 'mnist':
            assert args.arch in ['MLP', 'LeNet']
        elif args.dataset == 'cifar10':
            assert args.arch in ['ConvNet', 'NIN']
        else:
            raise NotImplementedError
        self.net = eval('%s()' % args.arch)
        self.net.load_weights()
        logger.info(self.net)

        # initialize inversenet
        self.inverse_net = eval('Inverse%s()' % args.arch)
        logger.info(self.inverse_net)
        self.inverse_net.copy_from(self.net)

        self.net.to(self.device)
        self.inverse_net.to(self.device)

        self.eps = 5e-6 if args.dataset == 'mnist' else 1e-5  # protect norm againse nan

    def net_forward(self, input_image):
        return self.net.forward(input_image.to(self.device))

    def inversenet_backward(self, input_image, idx):
        return self.inverse_net.forward_from_net(self.net, input_image, idx)

    def project_boundary_polyhedron(self, input_grad_, output_):
        batch_size = input_grad_.size()[0]  # e.g., 100 for mnist
        image_dim = input_grad_.size()[1]  # e.g., 784 for mnist
        # project under l_2 norm
        res_ = torch.abs(output_) / torch.norm(input_grad_ + self.eps, p=2, dim=1).view(output_.size())
        _, ii = torch.min(res_, 1)

        # dir_ = res_[np.arange(batch_size), ii.data].view(batch_size, 1)
        # advanced indexing seems to be buggy in pytorch 0.3.x, we use gather instead
        dir_ = res_.gather(1, ii.view(batch_size, 1))

        w = input_grad_.gather(
            2, ii.view(batch_size, 1, 1).expand(batch_size, image_dim, 1)).view(batch_size, image_dim)
        dir_ = dir_ * w / torch.norm(w + self.eps, p=2, dim=1).view(batch_size, 1)
        return dir_

    def forward_unlabel(self, input_image, label=None, pred=None, check=True):
        # this function is called when an image is correctly classified
        # label should be true label during training, and None during test

        num_image = input_image.size()[0]
        image_shape = input_image.size()
        self.label = pred.copy()
        if check:
            if self.training:
                # label should be true label
                assert label is not None
                m = self.label == label
                assert np.all(m.cpu().numpy())
            else:
                # label should be None
                assert label is None
        outputt = self.net_forward(input_image)
        idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
        output = outputt - outputt.gather(1, idx).expand_as(outputt)

        _, target_labels = torch.sort(-output, dim=1)
        target_labels = target_labels.data[:, :self.num_labels]

        ww = self.inversenet_backward(input_image, target_labels)
        w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        self.noises = dict()
        self.inputs_perturbed = dict()
        self.inputs_perturbed['step_0'] = input_image
        self.label_perturbed = self.label.copy()
        self.iteration = 0
        self.fooled = np.zeros(num_image).astype(np.bool)

        while True:
            self.iteration += 1
            noise_this_step = \
                self.project_boundary_polyhedron(w[:, :, 1:], output.gather(1, target_labels[:, 1:].to(self.device)))

            # if an image is already successfully fooled, no more perturbation should be applied to it
            t = torch.from_numpy(np.logical_not(self.fooled).astype(np.float32).copy()).to(self.device)
            t = t.view(num_image, 1).expand(num_image, noise_this_step.size()[1])
            self.noise_this_step = noise_this_step * t

            self.inputs_perturbed['step_%d' % self.iteration] = \
                self.inputs_perturbed['step_%d' % (self.iteration - 1)] + self.noise_this_step.view(image_shape)
            if len(self.noises) == 0:
                self.noises['step_%d' % self.iteration] = self.noise_this_step
            else:
                self.noises['step_%d' % self.iteration] = \
                    self.noises['step_%d' % (self.iteration - 1)] + self.noise_this_step

            # test whether we have successfully fooled these images
            _, t = torch.max(self.net_forward(
                input_image + (1 + self.overshot) * self.noises['step_%d' % self.iteration].view(image_shape)), 1)
            t = t.data.cpu().numpy().flatten()
            for i in range(num_image):
                # iterate over all images
                if not self.fooled[i]:
                    self.label_perturbed[i] = t[i]
                    if t[i] != self.label[i]:
                        self.fooled[i] = True

            if np.all(self.fooled):
                # quit if already fooled all images
                break
            if self.iteration == self.max_iter:
                # quit if max iteration
                break
            # if not quit, prepare the next fooling iteration

            outputt = self.net_forward(self.inputs_perturbed['step_%d' % self.iteration])
            idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
            output = outputt - outputt.gather(1, idx).expand_as(outputt)

            ww = self.inversenet_backward(self.inputs_perturbed['step_%d' % self.iteration], target_labels)
            w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        return (1 + self.overshot) * self.noises['step_%d' % self.iteration]

    def forward_correct(self, input_image, label=None, pred=None, check=True):
        # this function is called when an image is correctly classified
        # label should be true label during training, and None during test

        num_image = input_image.size()[0]
        image_shape = input_image.size()
        self.label = pred.copy()
        if check:
            if self.training:
                # label should be true label
                assert label is not None
                m = self.label == label
                assert np.all(m.cpu().numpy())
            else:
                # label should be None
                assert label is None
        outputt = self.net_forward(input_image)
        idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
        output = outputt - outputt.gather(1, idx).expand_as(outputt)

        _, target_labels = torch.sort(-output, dim=1)
        target_labels = target_labels.data[:, :self.num_labels]

        ww = self.inversenet_backward(input_image, target_labels)
        w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        self.noises = dict()
        self.inputs_perturbed = dict()
        self.inputs_perturbed['step_0'] = input_image
        self.label_perturbed = self.label.copy()
        self.iteration = 0
        self.fooled = np.zeros(num_image).astype(np.bool)

        while True:
            self.iteration += 1
            noise_this_step = \
                self.project_boundary_polyhedron(w[:, :, 1:], output.gather(1, target_labels[:, 1:].to(self.device)))

            # if an image is already successfully fooled, no more perturbation should be applied to it
            t = torch.from_numpy(np.logical_not(self.fooled).astype(np.float32).copy()).to(self.device)
            t = t.view(num_image, 1).expand(num_image, noise_this_step.size()[1])
            self.noise_this_step = noise_this_step * t

            self.inputs_perturbed['step_%d' % self.iteration] = \
                self.inputs_perturbed['step_%d' % (self.iteration - 1)] + self.noise_this_step.view(image_shape)
            if len(self.noises) == 0:
                self.noises['step_%d' % self.iteration] = self.noise_this_step
            else:
                self.noises['step_%d' % self.iteration] = \
                    self.noises['step_%d' % (self.iteration - 1)] + self.noise_this_step

            # test whether we have successfully fooled these images
            _, t = torch.max(self.net_forward(
                input_image + (1 + self.overshot) * self.noises['step_%d' % self.iteration].view(image_shape)), 1)
            t = t.data.cpu().numpy().flatten()
            for i in range(num_image):
                # iterate over all images
                if not self.fooled[i]:
                    self.label_perturbed[i] = t[i]
                    if t[i] != self.label[i]:
                        self.fooled[i] = True

            if np.all(self.fooled):
                # quit if already fooled all images
                break
            if self.iteration == self.max_iter:
                # quit if max iteration
                break
            # if not quit, prepare the next fooling iteration

            outputt = self.net_forward(self.inputs_perturbed['step_%d' % self.iteration])
            idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
            output = outputt - outputt.gather(1, idx).expand_as(outputt)

            ww = self.inversenet_backward(self.inputs_perturbed['step_%d' % self.iteration], target_labels)
            w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        return (1 + self.overshot) * self.noises['step_%d' % self.iteration]

    def forward_wrong(self, input_image, label, pred, check=True):
        # this function is called when an image is incorrectly classified
        # this function is only called during test, and label is true label

        num_image = input_image.size()[0]
        image_shape = input_image.size()
        self.label = pred.copy()
        if check:
            assert self.training
            assert label is not None
            m = self.label != label
            assert np.all(m.cpu().numpy())

        idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
        outputt = self.net_forward(input_image)
        output = outputt - outputt.gather(1, idx).expand_as(outputt)

        target_labels = torch.from_numpy(np.vstack((self.label, label)).T).to(self.device)

        ww = self.inversenet_backward(input_image, target_labels)
        w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        self.noises = dict()
        self.inputs_perturbed = dict()
        self.inputs_perturbed['step_0'] = input_image
        self.label_perturbed = self.label.copy()
        self.iteration = 0
        self.fooled = np.zeros(num_image).astype(np.bool)

        while True:
            self.iteration += 1
            noise_this_step = \
                self.project_boundary_polyhedron(w[:, :, 1:], output.gather(1, target_labels[:, 1:].to(self.device)))

            t = torch.from_numpy(np.logical_not(self.fooled).astype(np.float32)).to(self.device)
            t = t.view(num_image, 1).expand(num_image, noise_this_step.size()[1])
            self.noise_this_step = noise_this_step * t

            self.inputs_perturbed['step_%d' % self.iteration] = \
                self.inputs_perturbed['step_%d' % (self.iteration - 1)] + self.noise_this_step.view(image_shape)
            if len(self.noises) == 0:
                self.noises['step_%d' % self.iteration] = self.noise_this_step
            else:
                self.noises['step_%d' % self.iteration] = \
                    self.noises['step_%d' % (self.iteration - 1)] + self.noise_this_step

            _, t = torch.max(self.net_forward(
                input_image + (1 + self.overshot) * self.noises['step_%d' % self.iteration].view(image_shape)), 1)
            t = t.data.cpu().numpy().flatten()
            for i in range(num_image):
                if not self.fooled[i]:
                    self.label_perturbed[i] = t[i]
                    if t[i] == label[i]:
                        self.fooled[i] = True

            if np.all(self.fooled):
                break
            if self.iteration == self.max_iter:
                break

            outputt = self.net_forward(self.inputs_perturbed['step_%d' % self.iteration])
            idx = torch.from_numpy(self.label).to(self.device).view(num_image, 1)
            output = outputt - outputt.gather(1, idx).expand_as(outputt)

            # target will change as fooling process goes on
            # this is different from forward_correct
            self.label = outputt.data.cpu().numpy().argmax(axis=1)
            target_labels = torch.from_numpy(np.vstack((self.label, label)).T).to(self.device)

            ww = self.inversenet_backward(self.inputs_perturbed['step_%d' % self.iteration], target_labels)
            w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        return (1 + self.overshot) * self.noises['step_%d' % self.iteration]

    def forward(self, input_image):
        # this function should only be used during test
        # in training, use forward_correct and forward_wrong instead
        assert not self.training
        return self.forward_correct(input_image, check=False)
