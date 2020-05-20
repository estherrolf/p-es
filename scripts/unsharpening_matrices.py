import numpy as np
from numba import jit
import math
import scipy

from skimage.measure import compare_ssim as ssim
from sklearn.metrics.pairwise import euclidean_distances

# this file contains a bunch of ways to creat smoothing/distance matrices.
def Laplace_efficient(latlons_train, latlons_test, gamma):
    '''
    Computes the 
    '''
    n = len(latlons_train) + len(latlons_test)
    k = len(latlons_test)
                  
    latlons = np.vstack((latlons_train, latlons_test))
    # fill in all of the non-diagonal elements
   
    L_out = fast_sub_L(latlons,n,k,gamma)
    
    # add in the diagonal
    for i in range(k):
        L_out[i,i] += np.sum(L_out[:,i])
            
    return L_out

#@jit()
def Laplace_full(latlons_train, latlons_test, gamma, normalized=False):
    #n = len(latlons_train) + len(latlons_test)
    #k = len(latlons_test)
                  
    latlons = np.vstack((latlons_train, latlons_test))   
    D = RBF_matrix(latlons, gamma)
    L = Laplace_matrix(D,normalized=normalized)
    
    return L

@jit(nopython=True)                 
def fast_sub_L(latlons,n,k,gamma):
    L_out = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            d = (latlons[i,0] - latlons[j,0])**2 + (latlons[i,1] - latlons[j,1])**2 
            L_out[i,j] = - math.exp(-gamma*d)
    return L_out


def Laplace_matrix(W,normalized=False):
    D = np.diag(np.sum(W,axis=0))
    L = D - W
    if normalized:
        D_reg = np.diag(1/np.sqrt(np.sum(W,axis=0)))
        L = D_reg.dot(L.dot(D_reg))
    return L 


# get the distance matrix of samples. 
def distance_matrix_sq(z):
    # get the distance matrix of samples. 
    return euclidean_distances(z,z,squared=True)

@jit(nopython=True)
def distance_matrix(z):
    n = len(z)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            d = 0
            for k in range(len(z[0])):
                d += (z[i,k] - z[j,k])**2 
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix




#@jit(nopython=True)
#def distance_matrix_sqrt(latlons):
#    n = len(latlons)
#    dist_matrix = np.zeros((n,n))
#    for i in range(n):
#        for j in range(i+1, n):
#            d_sqrt = np.sqrt((latlons[i,0] - latlons[j,0])**2 + (latlons[i,1] - latlons[j,1])**2) 
##            dist_matrix[i][j] = d_sqrt
#            dist_matrix[j][i] = d_sqrt
#    return dist_matrix

@jit()
def RBF_matrix(latlons,gamma):
    dist_sq_matrix = distance_matrix(latlons)
    return fast_exp(-gamma * dist_sq_matrix)

#@jit()
#def RBF_matrix_sqrt(latlons,gamma):
#    dist_sq_matrix = distance_matrix_sqrt(latlons)
#    return fast_exp(-gamma/2.0 * dist_sq_matrix)


@jit(nopython=True)
#jit
def fast_exp(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = math.exp(X[i,j])
    return X

@jit()
def smooth_rbf_from_dist(dist_sq_matrix, y, gamma):
    y_smoothed = np.zeros(len(y))
    
    # create the RBF kernel matrix
    K = fast_exp(-gamma * dist_sq_matrix)
    # smoothed value for each point
    rel_weights = np.sum(K, axis=0)
    y_smoothed = K.dot(y) / rel_weights
    
    return y_smoothed



@jit()
def smooth_fixed_radius(dist_sq_matrix, y, r):
    y_smoothed = np.zeros(len(y))
    
    # create the RGB kernel matrix
    K = (dist_sq_matrix < r**2)
    # smoothed value for each point
    rel_weights = np.sum(K, axis=0)
    y_smoothed = K.dot(y) / rel_weights
    
    return y_smoothed

@jit()
def smooth_fixed_radius_with_rbf(dist_sq_matrix, y, r, gamma):
    y_smoothed = np.zeros(len(y))
    
    # create the RGB kernel matrix
    #K_r = (dist_sq_matrix < r**2)
    K = fast_exp(-gamma * dist_sq_matrix)
    
    K[dist_sq_matrix > r**2] = 0 
    # smoothed value for each point
    rel_weights = np.sum(K, axis=0)
    y_smoothed = K.dot(y) / rel_weights
    
    return y_smoothed

@jit()
def smooth_knn(dist_sq_matrix,y, k):
    y_smoothed = np.zeros(len(y))
    
    for i in range(len(y)):
        dists = dist_sq_matrix[i]
        all_neighbors = np.argsort(dists)
        # include yourself
        k_neighbors = all_neighbors[:k]
        total_sum = 0
        for idx in k_neighbors:
            total_sum += y[idx]
        y_smoothed[i] = total_sum / k
    return y_smoothed

@jit()
def smooth_knn_with_rbf(dist_sq_matrix,y, k, gamma):
    y_smoothed = np.zeros(len(y))
    
    K = fast_exp(-gamma * dist_sq_matrix)
    
    for i in range(len(y)):
        dists = dist_sq_matrix[i]
        all_neighbors = np.argsort(dists)
        # include yourself
        k_neighbors = all_neighbors[:k]
        total_sum = 0
        total_weight = 0
        for idx in k_neighbors:
            total_sum += y[idx]*K[i,idx]
            total_weight += K[i,idx]
        y_smoothed[i] = total_sum / total_weight
    return y_smoothed

def make_ssim_matrix(X):
    D_ssim = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(i,len(X)):
            inv_sim = 1 - ssim(X[i].reshape(8,8),X[j].reshape(8,8))
            D_ssim[i,j] = inv_sim
            D_ssim[j,i] = inv_sim
    return D_ssim

def sigma_to_gamma(sigma):
    return 0.5/(sigma**2)

def gamma_to_sigma(gamma):
    return np.sqrt(0.5/gamma)