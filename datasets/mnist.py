import torch.utils.data
import numpy as np
import scipy.io as sio


num_val = 10000  # first num_val examples in training set is used as validation set


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, phase='train'):
        imdb = sio.loadmat('data/mnist-data-0208ce21.mat')
        images = imdb['images'][0][0][0].transpose()
        sets = imdb['images'][0][0][3].flatten()
        labels = imdb['images'][0][0][2].flatten() - 1
        train_idx = np.where(sets == 1)[0][num_val:]
        val_idx = np.where(sets == 1)[0][:num_val]
        trainval_idx = np.where(sets == 1)[0]
        test_idx = np.where(sets == 3)[0]
        mean = imdb['images'][0][0][1].transpose()
        assert phase in ['train', 'val', 'trainval', 'test']
        self.images = eval('images[%s_idx]' % phase)
        self.labels = eval('labels[%s_idx]' % phase)
        self.mean = mean
        self.perm = np.arange(self.labels.size)

    def shuffle(self, perm):
        self.perm = perm

    def __getitem__(self, index):
        return self.images[self.perm[index]], self.labels[self.perm[index]]

    def __len__(self):
        return self.labels.size
