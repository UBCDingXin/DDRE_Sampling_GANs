"""
Some helpful functions

"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
from scipy.stats import multivariate_normal
import os


################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')
################################################################################
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

################################################################################
# torch dataset from numpy array
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        super(custom_dataset, self).__init__()

        self.data = data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.n_samples = self.data.shape[0]

    def __getitem__(self, index):

        x = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
        else:
            y = -1
        return x, y

    def __len__(self):
        return self.n_samples

################################################################################
# Plot training loss
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    plt.title('Loss')
    plt.savefig(filename)

################################################################################
# Creat sampler for a mixture of Gaussian distributions
def sampler_MixGaussian(n_samp, means, sigma = 0.05, dim = 2):
    '''

    n_samp: how many samples will be draw from this mixture Gaussian
    means: means for each components; a n_components by dim numpy array
    sigma: a fixed standard deviation
    dim: dimension of a component

    '''
    cov = np.diag(np.repeat(sigma**2, dim)) #covariance matrix; firxed for each component
    n_components = len(means) #number of components
    weights = np.ones(n_components, dtype=np.float64) / n_components
    membership = np.random.choice(np.arange(n_components), size = n_samp, p = weights)
    samples = np.zeros((1,dim))
    labels = np.zeros(1)
    for i in range(n_components):
        indx_current = np.where(membership==i)[0]
        nsamp_current = len(indx_current)
        mean_current = means[i]
        samples = np.concatenate((samples, np.random.multivariate_normal(mean_current, cov, size=nsamp_current)), axis=0)
        labels = np.concatenate((labels, np.repeat(i,nsamp_current)), axis=0)
    samples = samples[1:]
    labels = labels[1:]
    return samples, labels

################################################################################
# PDF for a mixture of Gaussian distributions

# def pdf_Gaussian(x, mean, cov):
#     #x and mean must be k by 1 vector
#     x = x.reshape(-1, 1); mean = mean.reshape(-1,1)
#     k = len(mean)
#     x_center = x-mean
#     tmp = np.matmul(np.matmul(np.transpose(x_center), np.linalg.inv(cov)), x_center)
#     pdf = 1/np.sqrt((2*np.pi)**k*np.linalg.det(cov))*np.exp(-1/2 * tmp)
#     return pdf

# def pdf_MixGaussian(x, means, sigma, mc=None):
#     # x: quantile
#     # means: means for all components; n_components by dim
#     # sigma: diagonal elements in the covariance matrix; a fixed cov for all components
#     # mc: mixture coefficients
#     (n_components, dim) = means.shape
#     cov = np.diag(np.repeat(sigma**2, dim))
#     if mc is None:
#         mc = np.ones(n_components)/n_components
#     assert np.sum(mc) == 1
#     pdf = 0
#     for i in range(n_components):
#         pdf += mc[i] * pdf_Gaussian(x, means[i], cov)
#     return pdf

def pdf_MixGaussian(X, means, sigma, mc=None):
    # X: quantile, n_samp by n_features
    # means: means for all components; n_components by n_features
    # sigma: diagonal elements in the covariance matrix; a fixed cov for all components
    # mc: mixture coefficients
    n_samp = X.shape[0]
    (n_components, n_features) = means.shape
    cov = np.diag(np.repeat(sigma**2, n_features))
    if mc is None:
        mc = np.ones(n_components)/n_components
    assert np.abs(np.sum(mc) - 1) < 1e-14
    #if np.sum(mc) != 1:
    #    print(np.sum(mc))
    pdfs = np.zeros(n_samp)
    for i in range(n_components):
        pdfs += mc[i] * multivariate_normal.pdf(X, mean=means[i], cov=cov)
    return pdfs



################################################################################
# Plot samples in a 2-D coordinate
def ScatterPoints(tar_samples, prop_samples, filename, plot_real_samples = False):
    # tar_samples and prop_samples are 2-D array: n_samples by num_features
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    plt.figure(figsize=(5, 5), facecolor='w')
    plt.grid(b=True)
    plt.scatter(tar_samples[:, 0], tar_samples[:, 1], c='blue', edgecolor='none', alpha=0.5)
    if not os.path.isfile(filename[0:-4]+'_realsamples.pdf') and plot_real_samples:
        plt.savefig(filename[0:-4]+'_realsamples.pdf')
    plt.scatter(prop_samples[:, 0], prop_samples[:, 1], c='g', edgecolor='none')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


################################################################################
# Train a MLP classifier


#adjust CNN learning rate
def adjust_learning_rate_PreNetDRE(optimizer, epoch, base_lr):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_lr
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_lr - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_PreNetDRE(epochs, trainloader, testloader, net, base_lr, optimizer, criterion, device="cuda"):
    net=net.to(device)
    for epoch in range(epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_PreNetDRE(optimizer, epoch, base_lr)
        for batch_idx, (batch_train_samples, batch_train_labels) in enumerate(trainloader):

            batch_train_samples = batch_train_samples.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.type(torch.long).to(device)

            #Forward pass
            outputs, _ = net(batch_train_samples)
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        test_acc = test_PreNetDRE(testloader, net, verbose=False)

        print('PreNetDRE: [epoch %d/%d] train_loss:%.3f, test_acc:%.3f' % (epoch+1, epochs, train_loss/(batch_idx+1), test_acc))
    #end for epoch

    return net, optimizer

def test_PreNetDRE(testloader, net, verbose=True, device="cuda"):
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for samples, labels in testloader:
            samples = samples.type(torch.float).to(device)
            labels = labels.type(torch.long).to(device)
            outputs, _ = net(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        if verbose:
            print('Test Accuracy of the model on the 10000 test samples: {} %'.format(100.0 * correct / total))
    return 100.0 * correct / total
