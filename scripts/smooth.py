import numpy as np
from numba import jit
import sklearn.metrics

import unsharpening_matrices
import utils

#@jit()
def smooth(S, y, c=None,normalize=True,normalize_diag=False, neighbor_cutoff=None):
    '''
    Args:
        S: (2d np.array of floats): smoothing matrix of dimension k x n
        y: (1d np.array of floats): label vector of dimension n
        normalize (bool): if True, normalizes weights 
    Returns: 
        y_smoothed: (1d np.array of floats): label vector resulting from smoothing y w.r.t. S
        
    Description: Smooths the vector y according
    '''

    # assert that the n dimensions match
    assert y.shape[0] == S.shape[1], "shapes don't match: y: {0}, S: {1}".format(y.shape,S.shape)
    if (normalize and normalize_diag):
        print("normalizing according to diagonal method")
    
    if len(y.shape) == 1:
        y = y.reshape(-1,1)
    
    if neighbor_cutoff is None:
        neighbor_cutoff = S.shape[1]
    
    # this comparsion b.c code hat uses this will put n if no neighbor cutoff
    S_this = S #.copy()
    
    if neighbor_cutoff < S_this.shape[1]:
        sorted_row_idxs = S_this.argsort(axis=1)
        k = neighbor_cutoff
        # only take values with k largest S values
        for i in range(S_this.shape[0]):
            S_this[i, sorted_row_idxs[i,:-k]] = 0 
        
    # smoothed value for each point
    if normalize_diag:
        rel_weights = np.sum(S_this, axis=1).reshape(-1,1)
        D_sqrt = np.diag(np.sqrt((1/rel_weights)).reshape(-1))
        S_this = D_sqrt.dot(S_this).dot(D_sqrt)
        if (rel_weights == 0).any():
            return np.zeros((S_this.shape[0], y.shape[1]))
        
    elif normalize:
        rel_weights = np.sum(S_this, axis=1).reshape(-1,1)
        if (rel_weights == 0).any():
            return np.zeros((S_this.shape[0], y.shape[1]))
        else:
            y_smoothed = S_this.dot(y) / rel_weights
    else:
        y_smoothed = S_this.dot(y)
    if c is not None:
        assert c >=0, "c not within range"
        assert c <= 1, "c not within range"
        y_smoothed = c*y_smoothed + (1-c)*y[:S_this.shape[0]]    
    return y_smoothed


def predict_and_smooth(X,y, D, train_idxs,test_idxs, gammas, model_function,
                       model_params=None, 
                       thresh=0.5, 
                       binary=True,
                       use_train_to_smooth=False):
    '''
    Args:
        X (n x d np array): feature matrix
        y (n x 1 np array): label vector
        D (n x n np array): distance matrix
        train_idxs (list of lists): train indices from which to run predictions. lists are nested by 
            num_tasks x trials_per_task
        test_idxs (list of lists): corresponding test from which to run predictions. lists are nested by 
            num_tasks x trials_per_task
        gammas (list of floats): gammas from which to define smoothing matrics
        model_function (function): model functior predictions
        model_params: input params for model function
        thresh (float or "cmn" or None): if we're threshholding on a scalar value, using cmn,
            or none of the above.
        binary (bool): are we doing binary classification?
    Returns:
    
    Description:
    
   '''
            
    # assume for now that every predictor has 1 hp -- TODO: generalize to many. 
    num_params = 1
    for k in model_params:
        num_params = num_params*len(model_params[k])
    
    unsmoothed_params_shape = (len(train_idxs),len(train_idxs[0]),num_params)
    # need an extra axis for which to represent gammas
    smoothed_params_shape = unsmoothed_params_shape + (len(gammas),)
    
    accs_unsmoothed = np.zeros(unsmoothed_params_shape)
    preds_unsmoothed = np.empty(unsmoothed_params_shape,dtype=np.ndarray)
    
    accs_smoothed = np.zeros(smoothed_params_shape)
    preds_smoothed = np.empty(smoothed_params_shape,dtype=np.ndarray)
    
    # fill in the predictions
    print(" on (of {0}): ".format(len(train_idxs)),end="")
    for i in range(len(train_idxs)):
        print(i,end=" ")
        print("[",end="")
        for j,(train_idx, test_idx) in enumerate(zip(train_idxs[i],test_idxs[i])):
            
            
            acc_vec = utils.define_vectorized_accuracy(y[test_idx],binary,thresh)
            
            print(j,end="")
            # predict
            predictions_this_split = model_function(X[train_idx],X[test_idx],y[train_idx],**model_params)
            accs_this_split = acc_vec(predictions_this_split)
            
            preds_unsmoothed[i,j] = predictions_this_split
            accs_unsmoothed[i,j] = accs_this_split
            
            # TODO: is there a faster way to do the smoothing
            #smooth
            for g,gamma in enumerate(gammas):
                if use_train_to_smooth:
                    test_then_train_idx = np.hstack((test_idx,train_idx))
                    S = np.array(unsharpening_matrices.fast_exp(-gamma*D[test_idx,:][:,test_then_train_idx]))
                    predictions_this_split_smoothed = smooth_many_vectors(predictions_this_split,
                                                                          S,postpend_train=y[train_idx])
                else:
                    S = np.array(unsharpening_matrices.fast_exp(-gamma*D[test_idx,:][:,test_idx]))
                    predictions_this_split_smoothed = smooth_many_vectors(predictions_this_split, S)
                accs_this_split_smoothed = acc_vec(predictions_this_split_smoothed)
             
                #record
                preds_smoothed[i,j,:,g] = predictions_this_split_smoothed
                accs_smoothed[i,j,:,g] = accs_this_split_smoothed
            
        print("]",end="")
    return accs_smoothed, accs_unsmoothed, preds_smoothed, preds_unsmoothed

def just_smooth(X,y, D, train_idxs,test_idxs, gammas,
                       thresh=0.5, 
                       binary=True):
    '''
    Args:
        X (n x d np array): feature matrix
        y (n x 1 np array): label vector
        D (n x n np array): distance matrix
        train_idxs (list of lists): train indices from which to run predictions. lists are nested by 
            num_tasks x trials_per_task
        test_idxs (list of lists): corresponding test from which to run predictions. lists are nested by 
            num_tasks x trials_per_task
        gammas (list of floats): gammas from which to define smoothing matrics
        model_function (function): model functior predictions
        model_params: input params for model function
        thresh (float or "cmn" or None): if we're threshholding on a scalar value, using cmn,
            or none of the above.
        binary (bool): are we doing binary classification?
    Returns:
    
    Description:
    
   '''
            
    # need an extra axis for which to represent gammas
    # 1 to replace the hp
    smoothed_params_shape = (len(train_idxs),len(train_idxs[0]),1,len(gammas))
    
    accs_smoothed = np.zeros(smoothed_params_shape)
    preds_smoothed = np.empty(smoothed_params_shape,dtype=np.ndarray)
    
    # fill in the predictions
    print(" on (of {0}): ".format(len(train_idxs)),end="")
    for i in range(len(train_idxs)):
        print(i,end=" ")
        print("[",end="")
        for j,(train_idx, test_idx) in enumerate(zip(train_idxs[i],test_idxs[i])):
            
            
            acc_fxn = utils.define_accuracy(y[test_idx],binary,thresh)
            
            print(j,end="")
            
            # TODO: is there a faster way to do the smoothing
            #smooth
            for g,gamma in enumerate(gammas):
                test_then_train_idx = np.hstack((test_idx,train_idx))
                S = np.array(unsharpening_matrices.fast_exp(-gamma*D[test_idx,:][:,train_idx]))
                
                predictions_this_split_smoothed = smooth(S,y[train_idx])
                accs_this_split_smoothed = acc_fxn(predictions_this_split_smoothed)
                #record
                preds_smoothed[i,j,0,g] = predictions_this_split_smoothed
                accs_smoothed[i,j,0,g] = accs_this_split_smoothed
            
        print("]",end="")
    return accs_smoothed, preds_smoothed


def smooth_many_vectors(unsmoothed_vectors, S,postpend_train=None): 
    '''
    Args:
        unsmoothed_vectors (np.ndarray): array of vectors to be smoothed
        S (np.ndarray): smoothing matrix to be applied to all of the vectos individually
    Returns;
        smoothed_vecs (np.ndarray): the result of smoothing the input vectors, in the same shape as
            the argument unsmoothed_vectors
    Description: smooths many vectors by one smoothing matrix S. 
    '''
    # TODO: Change this to vectorized form
    new_shape = unsmoothed_vectors.shape
    smoothed_vecs = np.empty(new_shape,dtype=np.ndarray)
    for i,vector in enumerate(unsmoothed_vectors):
        if postpend_train is None:
            smoothed_vecs[i] = smooth(S,vector)
        else:
            big_vector_smoothed = smooth(S,np.vstack((vector,postpend_train)))
            smoothed_vecs[i] = big_vector_smoothed[:len(vector)]
    return smoothed_vecs
    

            



                        
                        
    
