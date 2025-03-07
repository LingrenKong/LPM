import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import spatial
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from tqdm import tqdm

from data import load_data_from_pickle, save_test_data, save_train_data
from models import VGG, ResNet18
from utils import get_minibatches_idx, set_random_seed

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model(config):
    if config['model'] == 'ResNet18':
        model = ResNet18(color_channel=config['color_channel'])
    elif config['model'] == 'VGG11':
        model = VGG('VGG11', color_channel=config['color_channel'])
    elif config['model'] == 'VGG13':
        model = VGG('VGG13', color_channel=config['color_channel'])
    else:
        print('wrong model option')
        model = None
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'],  momentum=config['momentum'],
                          weight_decay=config['weight_decay'])

    return model, loss_function, optimizer

# FGSM attack code


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_step(model, device, loader, epsilon,tb,key):
    # Accuracy counter
    model.eval()
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(loader))
    type = key['type']
    tb.add_scalar(f'Accuracy/FGSM-{type}-epsilon={str(epsilon)}', final_acc, key['epoch'])
    #print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
    #                                                         correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def train_test_batch(trainloader,testloader,model,loss,optimizer,config,epsilon_list,tb):
    for epoch in tqdm(range(config['epoch_num'])):
        if epoch == int(config['epoch_num'] / 3):
            for g in optimizer.param_groups:
                g['lr'] = config['lr'] / 10
            print('divide current learning rate by 10')
        elif epoch == int(config['epoch_num'] * 2 / 3):
            for g in optimizer.param_groups:
                g['lr'] = config['lr'] / 100
            print('divide current learning rate by 10')
        model.train()
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_train_batch_size'],
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(
                np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(
                np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(
                1), Variable(targets.long().cuda()).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
        tb.add_scalar('Loss/train', total_loss, epoch)
        #test--------------
        model.eval()
        total_loss_test = 0
        minibatches_idx_test = get_minibatches_idx(len(testloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
        with torch.no_grad():
            for minibatch in minibatches_idx_test:
                inputs = torch.Tensor(
                    np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
                targets = torch.Tensor(
                    np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
                inputs, targets = Variable(inputs.cuda()).squeeze(
                    1), Variable(targets.long().cuda()).squeeze()
                outputs = model(inputs).squeeze()
                loss = loss_function(outputs, targets)
                total_loss_test += loss
        tb.add_scalar('Loss/test', total_loss_test, epoch)
        #FGSM
        if epoch%50 == 9:
            for eps in epsilon_list:
                key = {'type':'test','epoch':epoch}
                fgsm_step(model, device, testloader, eps,tb,key)
                #key = {'type':'train','epoch':epoch}
                #fgsm_step(model, device, trainloader, eps,tb,key)
        #ETF暂时没写





# old
def get_features(loader, model, config,type='test'):
    total_features = []
    total_labels = []
    minibatches_idx = get_minibatches_idx(len(loader), minibatch_size=config[f'simple_{type}_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(
            np.array([list(loader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(
            np.array([list(loader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(
            1), Variable(targets.cuda()).squeeze()
        features = model.get_features(inputs)
        total_features.extend(features.cpu().data.numpy().tolist())
        total_labels.extend(targets.cpu().data.numpy().tolist())
    total_features = np.array(total_features)
    total_labels = np.array(total_labels)
    print('total features', total_features.shape)
    print('total labels', total_labels.shape)
    avg_feature = np.mean(total_features, axis=0)
    # print('avg feature', np.linalg.norm(avg_feature))
    centralized_features = total_features - avg_feature
    feature_norm = np.square(np.linalg.norm(centralized_features, axis=1))
    class_features = []
    feature_norm_list = []
    for i in range(10):
        mask_index = (total_labels == i)
        mask_index = mask_index.reshape(len(mask_index), 1)
        # print('mask index', mask_index)
        if config['R'] == 'inf' and i == config['t1']:
            break
        class_features.append(
            np.sum(total_features * mask_index, axis=0) / np.sum(mask_index.reshape(-1)))
        feature_norm_list.append(
            np.sum(feature_norm * mask_index.reshape(-1)) / np.sum(mask_index.reshape(-1)))

    class_features = np.array(class_features)
    # print('original class features', class_features)
    class_features = np.array(class_features) - avg_feature
    # print('centralized class features', class_features)
    print('feature norm list', feature_norm_list)
    print('avg square feature norm', np.mean(feature_norm_list))
    return class_features

# old


def analyze_collapse(linear_weights, config, option='weights'):
    num_classes = len(linear_weights)
    weight_norm = [np.linalg.norm(linear_weights[i])
                   for i in range(num_classes)]
    cos_matrix = np.zeros((num_classes, num_classes))
    between_class_cos = []
    for i in range(num_classes):
        for j in range(num_classes):
            cos_value = 1 - \
                spatial.distance.cosine(linear_weights[i], linear_weights[j])
            cos_matrix[i, j] = cos_value
            if i != j:
                between_class_cos.append(cos_value)
    weight_norm = np.array(weight_norm)
    print('{0} avg square norm'.format(option),
          np.mean(np.square(weight_norm)))
    between_class_cos = np.array(between_class_cos)
    print('{0} norm'.format(option), weight_norm)
    print('cos {0} matrix'.format(option), cos_matrix)
    print('between class {0} cosine'.format(option), between_class_cos)
    print('std {0} norm over avg {0} norm'.format(option),
          np.std(weight_norm) / np.mean(weight_norm))
    print('avg between-class {0} cosine'.format(option),
          np.mean(between_class_cos))
    print('std between-class {0} cosine'.format(option),
          np.std(between_class_cos))
    print('avg {0} cosine to -1/(C-1)'.format(option),
          np.mean(np.abs(between_class_cos + 1 / (num_classes - 1))))
    # compute between-class cosine for small classes
    if config['t1'] != len(linear_weights):
        t1 = config['t1']
        print('{0} cosine for small classes'.format(
            option), cos_matrix[t1:, t1:])
        between_class_cos_small = []
        for i in range(10)[t1:]:
            for j in range(10)[t1:]:
                if i != j:
                    between_class_cos_small.append(cos_matrix[i, j])
        print(
            'between-calss {0} cosine for small classes'.format(option), between_class_cos_small)
        print('avg between-class {0} cosine for small classes'.format(
            option), np.mean(between_class_cos_small))
        print('std between-class {0} cosine for small classes'.format(
            option), np.std(between_class_cos_small))
        print('std {0} norm over avg {0} norm for small classes'.format(option), np.std(weight_norm[t1:]) /
              np.mean(weight_norm[t1:]))

# old


def analyze_dual(linear_weights, class_features):
    n_class = len(class_features)
    linear_weights = linear_weights[:n_class]
    linear_weights = linear_weights / np.linalg.norm(linear_weights)
    class_features = class_features / np.linalg.norm(class_features)
    # print('normalized linear weights', linear_weights)
    # print('normalized class features', class_features)
    print('dual distance', np.linalg.norm(linear_weights - class_features))
    print('dual distance square', np.square(
        np.linalg.norm(linear_weights - class_features)))


if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())

    # 初始化配置
    data_option = sys.argv[1].split('=')[1]
    model_option = sys.argv[2].split('=')[1]
    t1 = int(sys.argv[3].split('=')[1])
    R = (sys.argv[4].split('=')[1])
    config = {'dir_path': '/mnt/e/GitHub Repo/LPM', 'data': data_option, 'model': model_option, 't1': t1, 'R': R,
              'simple_train_batch_size': 512, 'simple_test_batch_size': 400, 'epoch_num': 360,#batch128,100
              'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'fixed': 'big'}
    # fixed: big/small
    if data_option == 'fashion_mnist':
        config['color_channel'] = 1
    else:
        config['color_channel'] = 3
    set_random_seed(666)
    if R == 'inf':
        config['big_class_sample_size'] = 5000
        config['small_class_sample_size'] = 0
    else:
        R = int(R)
        if data_option == 'cifar10':
            config['big_class_sample_size'] = 5000
            config['small_class_sample_size'] = 5000 // R
        elif data_option == 'fashion_mnist':
            config['big_class_sample_size'] = 6000
            config['small_class_sample_size'] = 6000 // R
        else:
            print('wrong data option')
    model_path = config['dir_path'] + '/models/' + config['data'] + '_' + config['model'] + '_t1=' + \
        str(config['t1']) + '_R=' + config['R'] + "_" + config['fixed'] + '.pt'
    from torch.utils.tensorboard import SummaryWriter
    model_name = config['data'] + '_' + config['model'] + '_t1=' + \
        str(config['t1']) + '_R=' + config['R'] + "_" + config['fixed']
    writer = SummaryWriter(comment=model_name)  # ./runs/ #全局变量

    print('save test data', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    set_random_seed(666)
    save_test_data(config)

    print('save train data', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    set_random_seed(666)
    save_train_data(config)

    set_random_seed(666)
    print('load data from pickle', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    train_data, test_data = load_data_from_pickle(config)

    print('build model', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    model, loss_function, optimizer = build_model(config)

    print('train model', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    epsilons = [0, .01, .02, .05, .1, .15, .2, .25, .3]
    use_cuda = True
    device = torch.device("cuda" if (
        use_cuda and torch.cuda.is_available()) else "cpu")
    train_test_batch(train_data, test_data, model, loss_function, optimizer, config, epsilons, writer)

    print('save model', 'time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    torch.save(model.state_dict(), model_path)

    print('end', 'time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


