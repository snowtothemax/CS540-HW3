from xml.dom.minidom import parseString
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, 0)
    return x


def get_covariance(dataset):
    return np.cov(dataset)

def get_eig(S, m):
    w,v = eigh(S,subset_by_index=[1024-m,1023])

    # flip ascending
    w = np.fliplr(w)
    w = np.flipud(w)
    v = np.flipud(v)
    return w,v


def get_eig_prop(S, prop):
    w,v = eigh(S)
    sum = np.sum(w)

    # new way to sort
    ind = np.argsort(w)[::-1]
    w = [ind]
    v = w[:,ind]

    # get values and sort
    w = w[w > sum * prop]
    inds = np.argwhere(w > sum * prop)
    inds = np.ndarray.flatten(inds)
    v = v[:,inds]

    return np.diag(w), v


def project_image(image, U):
    sums = []
    for j in range(np.size(U, 1)):
        #compute dot of row
        aij = np.dot(np.transpose(U[:,j]), image)
        #project row
        res = np.dot(aij, U[:,j])
        sums.append(res)

    return sum(sums)

def display_image(orig, proj):
    #reshape
    reshapedOrig = np.transpose(np.reshape(orig, (32,32)))
    reshapedProj = np.transpose(np.reshape(proj, (32,32)))

    #set gigure and axes
    fig, axs = plt.subplots(1, 2)

    #set titles
    axs[0].set_title('Original')
    axs[1].set_title('Projection')

    fig.colorbar(axs[0].imshow(reshapedOrig, aspect = 'equal'), ax=axs[0])
    fig.colorbar(axs[1].imshow(reshapedProj, aspect = 'equal'), ax=axs[1])

    plt.show()

x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)


