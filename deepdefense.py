#!/usr/bin/env python
import sys
import os
import os.path as osp
import glog as log
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets.mnist import MNISTDataset
from datasets.cifar10 import CIFAR10Dataset
from models.mnist import LeNet, InverseLeNet, MLP, InverseMLP
from models.cifar10 import ConvNet, InverseConvNet, NIN, InverseNIN


def parse_args():
    parser = argparse.ArgumentParser(description='Use DeepDefense to improve robustness')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='number of epochs to train')
    parser.add_argument('--max-iter', default=5, type=int,
                        help='max iteration in deepfool attack')
    parser.add_argument('--lmbd', default=15, type=float,
                        help='lmbd in regularization term')
    parser.add_argument('--c', default=25, type=float,
                        help='c in regularization term')
    parser.add_argument('--d', default=5, type=float,
                        help='d in regularization term')
    parser.add_argument('--decay', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--batch', default=100, type=int,
                        help='actual batch size in each iteration during training. '
                             'we use gradient accumulation if args.batch < args.train_batch')
    parser.add_argument('--train-batch', default=100, type=int,
                        help='training batch size. we always collect args.train_batch samples for one update')
    parser.add_argument('--test-batch', default=100, type=int,
                        help='test batch size')
    parser.add_argument('--exp-dir', default='output/debug', type=str,
                        help='directory to save models and logs for current experiment')
    parser.add_argument('--pretest', action='store_true',
                        help='evaluate model before training')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='which dataset to use, e.g., mnist or cifar10')
    parser.add_argument('--arch', default='LeNet', type=str,
                        help='network architecture, e.g., LeNet or MLP')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class DeepFool(nn.Module):
    def __init__(self):
        super(DeepFool, self).__init__()

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
        log.info(self.net)

        # initialize inversenet
        self.inverse_net = eval('Inverse%s()' % args.arch)
        log.info(self.inverse_net)
        self.inverse_net.copy_from(self.net)

        self.net.cuda()
        self.inverse_net.cuda()

        self.eps = 5e-6 if args.dataset == 'mnist' else 1e-5  # protect norm againse nan

    def net_forward(self, input_image):
        return self.net.forward(input_image.cuda())

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
                assert np.all(self.label == label)
            else:
                # label should be None
                assert label is None
        outputt = self.net_forward(input_image)
        idx = torch.from_numpy(self.label).cuda().view(num_image, 1)
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
                self.project_boundary_polyhedron(w[:, :, 1:], output.gather(1, target_labels[:, 1:].cuda()))

            # if an image is already successfully fooled, no more perturbation should be applied to it
            t = torch.from_numpy(np.logical_not(self.fooled).astype(np.float32).copy()).cuda()
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
            idx = torch.from_numpy(self.label).cuda().view(num_image, 1)
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
            assert np.all(self.label != label)

        idx = torch.from_numpy(self.label).cuda().view(num_image, 1)
        outputt = self.net_forward(input_image)
        output = outputt - outputt.gather(1, idx).expand_as(outputt)

        target_labels = torch.from_numpy(np.vstack((self.label, label)).T).cuda()

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
                self.project_boundary_polyhedron(w[:, :, 1:], output.gather(1, target_labels[:, 1:].cuda()))

            t = torch.from_numpy(np.logical_not(self.fooled).astype(np.float32)).cuda()
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
            idx = torch.from_numpy(self.label).cuda().view(num_image, 1)
            output = outputt - outputt.gather(1, idx).expand_as(outputt)

            # target will change as fooling process goes on
            # this is different from forward_correct
            self.label = outputt.data.cpu().numpy().argmax(axis=1)
            target_labels = torch.from_numpy(np.vstack((self.label, label)).T).cuda()

            ww = self.inversenet_backward(self.inputs_perturbed['step_%d' % self.iteration], target_labels)
            w = ww - ww[:, :, 0].contiguous().view(ww.size()[0], ww.size()[1], 1).expand_as(ww)

        return (1 + self.overshot) * self.noises['step_%d' % self.iteration]

    def forward(self, input_image):
        # this function should only be used during test
        # in training, use forward_correct and forward_wrong instead
        assert not self.training
        return self.forward_correct(input_image, check=False)


def test(model, phases='test'):
    model.eval()
    result = dict()
    if isinstance(phases, str):
        phases = [phases]
    for phase in phases:
        log.info('Evaluating deepfool robustness, phase=%s' % phase)
        loader = eval('%s_loader' % phase)

        num_image = len(loader.dataset)
        assert num_image % len(loader) == 0
        log.info('Found %d images' % num_image)

        accuracy = np.zeros(num_image)
        ce_loss = np.zeros(num_image)
        noise_norm = np.zeros(num_image)
        ratio = np.zeros(num_image)
        iteration = np.zeros(num_image)

        for index, (image, label) in enumerate(loader):
            # get one batch
            image_var = image.cuda()
            image_var.requires_grad = True
            label_var = label.long().cuda()
            selected = np.arange(index * args.test_batch, (index + 1) * args.test_batch)

            # calculate cross entropy
            forward_result_var = model.net(image_var)
            ce_loss_var = F.cross_entropy(forward_result_var, label_var)
            ce_loss[selected] = ce_loss_var.data.cpu().numpy()
            pred = forward_result_var.data.cpu().numpy().argmax(axis=1)

            # calculate accuracy
            accuracy[selected] = pred == label

            # calculate perturbation norm
            noise_var = model.forward_correct(image_var, label=label.cpu().numpy(), pred=pred, check=False)
            noise_loss_var = torch.norm(noise_var, dim=1)
            noise_norm[selected] = noise_loss_var.data.cpu().numpy().flatten()

            # calculate ratio
            # l_2 norm
            t = torch.norm(image_var.view(args.test_batch, -1), dim=1).data.cpu().numpy().flatten()
            ratio[selected] = noise_norm[selected] / t

            # save number of iteration
            iteration[selected] = model.iteration

            n = (index + 1) * args.test_batch
            if n % 1000 == 0:
                log.info('Evaluating %s set %d / %d,' % (phase, n, num_image))
                log.info('\tnoise_norm\t: %f' % (noise_norm.sum() / n))
                log.info('\tratio\t\t: %f' % (ratio.sum() / n))
                log.info('\tce_loss\t\t: %f' % (ce_loss.sum() / n))
                log.info('\taccuracy\t: %f' % (accuracy.sum() / n))
                log.info('\titeartion\t: %f' % (iteration.sum() / n))

        log.info('Performance on %s set is:' % phase)
        log.info('\tnoise_norm\t: %f' % noise_norm.mean())
        log.info('\tratio\t\t: %f' % ratio.mean())
        log.info('\tce_loss\t\t: %f' % ce_loss.mean())
        log.info('\taccuracy\t: %f' % accuracy.mean())

        result['%s_accuracy' % phase] = accuracy.mean()
        result['%s_ratio' % phase] = ratio.mean()

    log.info('Performance of current model is:')
    for phase in ['train', 'val', 'test']:
        if '%s_accuracy' % phase in result:
            log.info('\t%s accuracy\t: %f' % (phase, result['%s_accuracy' % phase]))
            log.info('\t%s ratio\t: %f' % (phase, result['%s_ratio' % phase]))


def train(model):
    num_epoch = args.epochs
    optimizer = optim.SGD([
        {'params': [p[1] for p in list(model.named_parameters())[1::2]], 'lr': args.lr, 'weight_decay': 0,
         'momentum': 0.9},  # bias
        {'params': [p[1] for p in list(model.named_parameters())[::2]], 'lr': args.lr, 'weight_decay': args.decay,
         'momentum': 0.9}  # weight
    ])
    num_image = len(train_loader.dataset)
    log.info('Found %d images' % num_image)

    assert (args.train_batch % args.batch == 0) and (args.train_batch >= args.batch)

    for epoch_idx in range(num_epoch):
        log.info('Training for %d epoch' % epoch_idx)
        model.zero_grad()

        # reduce learning
        if epoch_idx == (0.8 * args.epochs):
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                new_lr = lr * 0.5
                param_group['lr'] = new_lr
            log.info('epoch %d, cut learning rate from %f to %f' % (epoch_idx, lr, new_lr))

        perm = np.random.permutation(num_image)
        train_loader.dataset.shuffle(perm)
        for index, (image, label) in enumerate(train_loader):
            model.train()
            batch_in_train_batch = index % (args.train_batch // args.batch)
            if batch_in_train_batch == 0:
                noise_norm = np.zeros(args.train_batch)
                ratio = np.zeros(args.train_batch)
                ce_loss = np.zeros(args.train_batch)
                loss = np.zeros(args.train_batch)
                accuracy = np.zeros(args.train_batch)
                grad_norm = np.zeros(args.train_batch)
                optimizer.zero_grad()

            # get one batch data
            if (args.dataset == 'cifar10') and (args.arch == 'NIN') and (np.random.rand() < 0.5):
                # flip with a probability of 50%
                inv = torch.arange(image.size(2) - 1, -1, -1).long()
                image = image.index_select(2, inv)
            image_var = image.cuda()
            image_var.requires_grad = True
            label_var = label.long().cuda()
            # selected index in train batch, used to store ce_loss and loss
            selected_in_train_batch = np.arange(batch_in_train_batch * args.batch,
                                                (batch_in_train_batch + 1) * args.batch).astype(np.int)

            # split pos and neg
            forward_result_var = model.net(image_var)
            _, pred = torch.max(forward_result_var, 1)
            pred = pred.data.cpu().numpy().flatten()
            pos_idx = np.where(pred == label)[0]
            neg_idx = np.where(pred != label)[0]
            accuracy[selected_in_train_batch] = pred == label

            # adversarial training
            ce_loss_var = F.cross_entropy(forward_result_var, label_var)
            ce_loss_var = ce_loss_var * args.batch / args.train_batch
            ce_loss_var.backward(retain_graph=True)
            ce_loss[selected_in_train_batch] = ce_loss_var.data.cpu().numpy()

            if (args.lmbd > 0) and (pos_idx.size > 0):
                pos_idx_var = torch.from_numpy(pos_idx).cuda()
                pos_image = image_var.index_select(0, pos_idx_var)
                noise_var = model.forward_correct(input_image=pos_image,
                                                  label=label[pos_idx],
                                                  pred=pred[pos_idx],
                                                  check=True)
                noise_norm[batch_in_train_batch * args.batch + pos_idx] = torch.norm(noise_var,
                                                                                     dim=1).data.cpu().numpy().flatten()
                # l_2 norm
                ratio[batch_in_train_batch * args.batch + pos_idx] = \
                    noise_norm[batch_in_train_batch * args.batch + pos_idx] / \
                    torch.norm(pos_image.view(pos_idx.size, -1), dim=1).data.cpu().numpy().flatten()

                # calculate perturbation norm
                noise_loss_var = torch.norm(noise_var, dim=1)
                t = pos_image.view(pos_idx.size, -1)
                noise_loss_var = noise_loss_var / torch.norm(t, dim=1)

                loss_var = args.lmbd * torch.exp(-args.c * noise_loss_var)
                loss_var = loss_var.sum()
                loss[batch_in_train_batch * args.batch + pos_idx] = loss_var.data.cpu().numpy() / args.batch

                # BP
                loss_var = loss_var / args.train_batch
                loss_var.backward()

            if (args.lmbd > 0) and (neg_idx.size > 0):
                neg_idx_var = torch.from_numpy(neg_idx).cuda()
                neg_image = image_var.index_select(0, neg_idx_var)
                noise_var = model.forward_wrong(input_image=neg_image,
                                                label=label[neg_idx],
                                                pred=pred[neg_idx],
                                                check=True)
                noise_norm[batch_in_train_batch * args.batch + neg_idx] = \
                    torch.norm(noise_var, dim=1).data.cpu().numpy().flatten()

                # l_2 norm
                ratio[batch_in_train_batch * args.batch + neg_idx] = \
                    noise_norm[batch_in_train_batch * args.batch + neg_idx] /\
                    torch.norm(neg_image.view(neg_idx.size, -1), dim=1).data.cpu().numpy().flatten()

                # calculate perturbation norm
                noise_loss_var = torch.norm(noise_var, dim=1)
                t = neg_image.view(neg_idx.size, -1)
                noise_loss_var = noise_loss_var / torch.norm(t, dim=1)

                loss_var = args.lmbd * torch.exp(args.d * noise_loss_var)
                loss_var = loss_var.sum()
                loss[batch_in_train_batch * args.batch + neg_idx] = loss_var.data.cpu().numpy() / args.batch

                # BP
                loss_var = loss_var / args.train_batch
                loss_var.backward()

            # calculate grad norm
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm[selected_in_train_batch] += p.grad.data.norm(2) ** 2
            grad_norm[selected_in_train_batch] = np.sqrt(grad_norm[selected_in_train_batch])

            # update weights
            if batch_in_train_batch == (args.train_batch / args.batch - 1):
                optimizer.step()
                optimizer.zero_grad()

                log.info('Processing %d - %d / %d' % ((index + 1) * args.batch - args.train_batch,
                                                      (index + 1) * args.batch, num_image))
                log.info('\tnoise_norm\t: %f' % noise_norm.mean())
                log.info('\tgrad_norm\t: %f' % grad_norm.mean())
                log.info('\tratio\t\t: %f' % ratio.mean())
                log.info('\tce_loss\t\t: %f' % ce_loss.mean())
                log.info('\tloss\t\t: %f' % loss.mean())
                log.info('\taccuracy\t: %f' % accuracy.mean())

        # evaluate and save model after each epoch
        log.info('Evaluating model after epoch %d' % epoch_idx)
        test(model, phases='test')

        # save model
        fname = osp.join(args.exp_dir, 'epoch_%d.model' % epoch_idx)
        if not osp.exists(osp.dirname(fname)):
            os.makedirs(osp.dirname(fname))
        torch.save(model.state_dict(), fname)
        log.info('Model of epoch %d saved to %s' % (epoch_idx, fname))


def main():
    model = DeepFool()

    if args.pretest:
        log.info('Evaluating performance before fine-tune')
        test(model, phases='test')

    log.info('Fine-tuning network')
    train(model)

    log.info('Saving model')
    fname = osp.join(args.exp_dir, 'final.model')
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
    torch.save(model.cpu().state_dict(), fname)

    log.info('Final model saved to %s' % fname)


if __name__ == '__main__':
    args = parse_args()

    log.info('Called with args:')
    log.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(MNISTDataset(phase='trainval'),
                                                   batch_size=args.batch, shuffle=False, num_workers=4,
                                                   pin_memory=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(MNISTDataset(phase='test'),
                                                  batch_size=args.test_batch, shuffle=False, num_workers=4,
                                                  pin_memory=False, drop_last=False)
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(CIFAR10Dataset(phase='trainval'),
                                                   batch_size=args.batch, shuffle=False, num_workers=4,
                                                   pin_memory=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(CIFAR10Dataset(phase='test'),
                                                  batch_size=args.test_batch, shuffle=False, num_workers=4,
                                                  pin_memory=False, drop_last=False)
    else:
        raise NotImplementedError

    # print this script to log
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    with open(fname, 'r') as f:
        log.info(f.read())

    # make experiment directory
    if not osp.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # dump config
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    os.system('cp %s %s' % (fname, args.exp_dir))
    os.system('cp -r datasets models %s' % args.exp_dir)

    # do the business
    main()
