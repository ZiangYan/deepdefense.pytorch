import torch.utils.data
import numpy as np


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, phase='train', num_val=5000):
        import scipy.io as sio
        imdb = sio.loadmat('data/cifar10-data-ce5d97dd.mat')
        images = imdb['images'][0][0][0].transpose()
        sets = imdb['images'][0][0][2].flatten()
        labels = (imdb['images'][0][0][1].flatten() - 1).astype(np.int64)
        train_idx = np.where(sets == 1)[0][num_val:]
        val_idx = np.where(sets == 1)[0][:num_val]
        trainval_idx = np.where(sets == 1)[0]
        test_idx = np.where(sets == 3)[0]
        assert phase in ['train', 'val', 'trainval', 'test']
        self.images = eval('images[%s_idx]' % phase)
        self.labels = eval('labels[%s_idx]' % phase)
        self.perm = np.arange(self.labels.size)

    def __getitem__(self, index):
        if np.random.rand() > 0.5:
            images = np.fliplr(self.images[self.perm[index]]).copy()
        else:
            images = self.images[self.perm[index]]
        return images, self.labels[self.perm[index]]

    def __len__(self):
        return self.labels.size

    def shuffle(self, perm):
        self.perm = perm
