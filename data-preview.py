from data import load_data_from_pickle
import numpy as np
conf1 = {'dir_path':'/mnt/e/GitHub Repo/LPM','data':'fashion_mnist','model':'VGG13','t1': '10', 'R': '1',
              'simple_test_batch_size': 100, 'fixed': 'big', 'weight_decay': 5e-4}

train_data, test_data = load_data_from_pickle(conf1)

data, target = test_data[0]
print(data.detach().cpu().numpy().shape)
print(target.detach().cpu().numpy().shape)
print(data.detach().cpu().numpy())
print(target.detach().cpu().numpy())

tt = data.detach().cpu().numpy()
print(np.max(tt),np.min(tt))