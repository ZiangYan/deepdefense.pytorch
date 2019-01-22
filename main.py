import os
import sys
import json
import time
import socket
import random
import logging
import argparse
import os.path as osp
from datetime import datetime

from tensorboardX import SummaryWriter

from datasets.mnist import MNISTDataset
from datasets.cifar10 import CIFAR10Dataset

from deepdefense import *


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
writer = None
device = "cpu"


def parse_args():
    global logger, writer, device
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
    parser.add_argument('--arch', default='LeNet', type=str, help='network architecture, e.g., LeNet or MLP')
    parser.add_argument('--gpu-id', type=str, default="1", metavar='N', help='gpu id list (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    current_time = datetime.now().strftime('%m%d%H%M%S')
    exp_seed = random.randrange(sys.maxsize) % 1000

    args.seed = int(current_time)
    dir_path = os.path.join(os.environ['HOME'], 'project/runs', args.dataset + "_" + args.arch + "_" + current_time)
    logfile = "log/%s_%s_%d_%s_%d.log" % (args.dataset, args.arch, args.epochs, current_time, exp_seed)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    writer = SummaryWriter(log_dir=dir_path)
    return args


def test(epoch_idx, model, phases='test'):
    global device
    model.eval()
    result = dict()
    if isinstance(phases, str):
        phases = [phases]
    for phase in phases:
        logger.info('Evaluating deepfool robustness, phase=%s' % phase)
        loader = eval('%s_loader' % phase)

        num_image = len(loader.dataset)
        assert num_image % len(loader) == 0
        logger.info('Found %d images' % num_image)

        accuracy = np.zeros(num_image)
        ce_loss = np.zeros(num_image)
        noise_norm = np.zeros(num_image)
        ratio = np.zeros(num_image)
        iteration = np.zeros(num_image)

        for index, (image, label) in enumerate(loader):
            # get one batch
            image_var = image.to(device)
            image_var.requires_grad = True
            label_var = label.long().to(device)
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
            # if n % 1000 == 0:
            #     logger.info('Evaluating %s set %d / %d,' % (phase, n, num_image))
            #     logger.info('\tnoise_norm\t: %f' % (noise_norm.sum() / n))
            #     logger.info('\tratio\t\t: %f' % (ratio.sum() / n))
            #     logger.info('\tce_loss\t\t: %f' % (ce_loss.sum() / n))
            #     logger.info('\taccuracy\t: %f' % (accuracy.sum() / n))
            #     logger.info('\titeartion\t: %f' % (iteration.sum() / n))

        logger.info('Performance on %s set is:' % phase)
        logger.info('\tnoise_norm\t: %f' % noise_norm.mean())
        logger.info('\tratio\t\t: %f' % ratio.mean())
        logger.info('\tce_loss\t\t: %f' % ce_loss.mean())
        logger.info('\taccuracy\t: %f' % accuracy.mean())

        result['%s_accuracy' % phase] = accuracy.mean()
        result['%s_ratio' % phase] = ratio.mean()

    logger.info('Performance of current model is:')
    for phase in ['train', 'val', 'test']:
        if '%s_accuracy' % phase in result:
            print('\t%s accuracy\t: %f' % (phase, result['%s_accuracy' % phase]))
            print('\t%s ratio\t: %f' % (phase, result['%s_ratio' % phase]))
            writer.add_scalar("%s/Accuracy" % phase, result['%s_accuracy' % phase], epoch_idx)
            writer.add_scalar("%s/ratio" % phase, result['%s_ratio' % phase], epoch_idx)


def deep_defense_reg(model, images_var, label, pred, idx, lmbd=1, class_coef=1, correct=True):
    global device
    idx_var = torch.from_numpy(idx).to(device)
    images = images_var.index_select(0, idx_var)
    if correct:
        noise_var = model.forward_correct(input_image=images, label=label[idx], pred=pred[idx], check=True)
    else:
        noise_var = model.forward_wrong(input_image=images, label=label[idx], pred=pred[idx], check=True)

    # calculate perturbation norm
    noise_loss_var = torch.norm(noise_var, dim=1)
    t = images.view(idx.size, -1)
    noise_loss_var = noise_loss_var / torch.norm(t, dim=1)

    loss_var = lmbd * torch.exp(class_coef * noise_loss_var)
    loss_var = loss_var.sum()
    return loss_var, noise_var, images


def train(model):
    global device
    num_epoch = args.epochs
    optimizer = optim.SGD([
        {'params': [p[1] for p in list(model.named_parameters())[1::2]], 'lr': args.lr, 'weight_decay': 0,
         'momentum': 0.9},  # bias
        {'params': [p[1] for p in list(model.named_parameters())[::2]], 'lr': args.lr, 'weight_decay': args.decay,
         'momentum': 0.9}  # weight
    ])
    num_image = len(train_loader.dataset)
    logger.info('Found %d images' % num_image)

    assert (args.train_batch % args.batch == 0) and (args.train_batch >= args.batch)

    for epoch_idx in range(num_epoch):
        logger.info('Training for %d epoch' % epoch_idx)
        model.zero_grad()

        # reduce learning
        if epoch_idx == (0.8 * args.epochs):
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                new_lr = lr * 0.5
                param_group['lr'] = new_lr
            logger.info('epoch %d, cut learning rate from %f to %f' % (epoch_idx, lr, new_lr))

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
            image_var = image.to(device)
            image_var.requires_grad = True
            label_var = label.long().to(device)
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
            # args.batch , actual batch size(the last batch maybe truncated)
            ce_loss_var = ce_loss_var * args.batch / args.train_batch
            ce_loss_var.backward(retain_graph=True)
            ce_loss[selected_in_train_batch] = ce_loss_var.data.cpu().numpy()

            if (args.lmbd > 0) and (pos_idx.size > 0):
                idx = pos_idx
                loss_var, noise_var, images = deep_defense_reg(model, image_var, label, pred, idx, args.lmbd, -args.c, True)

                # l_2 norm
                noise_norm[batch_in_train_batch * args.batch + idx] = torch.norm(noise_var, dim=1).data.cpu().numpy().flatten()
                ratio[batch_in_train_batch * args.batch + idx] = noise_norm[batch_in_train_batch * args.batch + idx] / torch.norm(images.view(idx.size, -1), dim=1).data.cpu().numpy().flatten()
                loss[batch_in_train_batch * args.batch + idx] = loss_var.data.cpu().numpy() / args.batch
                # BP
                loss_var = loss_var / args.train_batch
                loss_var.backward()

            if (args.lmbd > 0) and (neg_idx.size > 0):
                idx = neg_idx
                loss_var, noise_var, images = deep_defense_reg(model, image_var, label, pred, idx, args.lmbd, args.d, False)

                # l_2 norm
                noise_norm[batch_in_train_batch * args.batch + idx] = torch.norm(noise_var, dim=1).data.cpu().numpy().flatten()
                ratio[batch_in_train_batch * args.batch + idx] = noise_norm[batch_in_train_batch * args.batch + idx] / torch.norm(images.view(idx.size, -1), dim=1).data.cpu().numpy().flatten()
                loss[batch_in_train_batch * args.batch + idx] = loss_var.data.cpu().numpy() / args.batch
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

                if index % 100 == 0:
                    logger.info('Processing %d - %d / %d' % ((index + 1) * args.batch - args.train_batch,
                                                          (index + 1) * args.batch, num_image))
                    logger.info('\tnoise_norm\t: %f' % noise_norm.mean())
                    logger.info('\tgrad_norm\t: %f' % grad_norm.mean())
                    logger.info('\tratio\t\t: %f' % ratio.mean())
                    logger.info('\tce_loss\t\t: %f' % ce_loss.mean())
                    logger.info('\tloss\t\t: %f' % loss.mean())
                    logger.info('\taccuracy\t: %f' % accuracy.mean())

        # evaluate and save model after each epoch
        print('Evaluating model after epoch %d' % epoch_idx)
        test(epoch_idx+1, model, phases='test')

        # save model
        # fname = osp.join(args.exp_dir, 'epoch_%d.model' % epoch_idx)
        # if not osp.exists(osp.dirname(fname)):
        #     os.makedirs(osp.dirname(fname))
        # torch.save(model.state_dict(), fname)
        # print('Model of epoch %d saved to %s' % (epoch_idx, fname))


def main():
    model = DeepFool(args)

    if args.pretest:
        logger.info('Evaluating performance before fine-tune')
        test(0, model, phases='test')

    logger.info('Fine-tuning network')
    train(model)

    logger.info('Saving model')
    fname = osp.join(args.exp_dir, 'final.model')
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
    torch.save(model.cpu().state_dict(), fname)

    logger.info('Final model saved to %s' % fname)


if __name__ == '__main__':
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

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

    # make experiment directory
    if not osp.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # dump config
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    args.device = device

    # do the business
    main()
