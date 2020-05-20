import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import sklearn.metrics

def fetch_20_newsgroups(cat,subset="train",n_pca_components=10):
    print("fetching data")
    newsgroups_data = fetch_20newsgroups(subset=subset, categories=cat,remove=('headers', 'footers', 'quotes'))
    newsgroups_data_extra = fetch_20newsgroups(subset=subset, categories=cat) 
    vectorizer = TfidfVectorizer()
    #featuress
    X = vectorizer.fit_transform(newsgroups_data.data).todense().astype('float64')
    # "instrinsic space" - still not sure PCA on this is really the thing to do here
    Z_big = vectorizer.fit_transform(newsgroups_data_extra.data).todense().astype('float64')
    #n_pca_components = 3
    print("fitting pca with n_components {0}".format(n_pca_components))
    this_pca = PCA(n_components=n_pca_components)
    this_pca.fit(Z_big)
    Z_pca = this_pca.transform(Z_big)
    
    #labels
    y = np.array(newsgroups_data.target).reshape(-1,1).astype('float64')
    
   
    return X,Z_pca,y

def split_pandas_to_xyz(df, x_columns, y_columns, z_columns, n_train,n_val, n_holdout, log_y=True, rs=0):
    rs = np.random.RandomState(seed=rs)
    n = n_train+n_val+n_holdout
    
    random_idxs = rs.choice(df.shape[0], n,replace=False).astype(int)
    
    # randomize here, then index as chunks to split sets
    df_subset = df.iloc[random_idxs]
    X = df_subset[x_columns].values.astype('float')
    if log_y:
        y = np.log(df_subset[y_columns].values.astype('float'))
    else:
        y = df_subset[y_columns].values.astype('float')
    z = df_subset[z_columns].values.astype('float')

    # split into train, validation, holdout sets
    train_idxs = np.arange(n_train)
    val_idxs = np.arange(n_train,n_train+n_val)
    holdout_idxs = np.arange(n_train+n_val, n_train+n_val+n_holdout)

    X_train, X_val, X_holdout = X[train_idxs],X[val_idxs],X[holdout_idxs]
    y_train, y_val, y_holdout = y[train_idxs],y[val_idxs],y[holdout_idxs]
    z_train, z_val, z_holdout = z[train_idxs],z[val_idxs],z[holdout_idxs]
    return {"X_train":X_train,"X_val":X_val,"X_holdout":X_holdout,
            "y_train":y_train,"y_val":y_val,"y_holdout":y_holdout,
            "z_train":z_train,"z_val":z_val,"z_holdout":z_holdout}

def pandas_to_xyz_no_splits(df, x_columns, y_columns, z_columns, n, log_y=True, rs=0):
    rs = np.random.RandomState(seed=rs)    
    # randomize here 
    random_idxs = rs.choice(df.shape[0], n,replace=False).astype(int)

    df_subset = df.iloc[random_idxs]
    X = df_subset[x_columns].values.astype('float')
    if log_y:
        y = np.log(df_subset[y_columns].values.astype('float'))
    else:
        y = df_subset[y_columns].values.astype('float')
    z = df_subset[z_columns].values.astype('float')

    X = X[:n]
    y = y[:n]
    z = z[:n]
    
    return {"X":X, "y":y, "z":z}



def get_variable_train_sets_with_constant_tests(total_samples, num_folds, num_trains):
    '''
    get the indices for experiments :) 
    '''
    # if we're keeping train size fixed, max it out
    if num_trains == None:
        num_trains = [int(np.floor((num_folds-1)/num_folds *total_samples))]
        
    assert max(num_trains) + total_samples/num_folds <= total_samples, "not enough data to satisfy"
        
    rs = np.random.RandomState(seed=0)    
    kf = KFold(n_splits=num_folds,random_state=0,shuffle=False)
    # dummy X
    X = np.arange(total_samples)
    
    train_idxs = []
    test_idxs = []
        
    for num_train in num_trains:
        train_idxs_0 = []
        test_idxs_0 = []
        for train_index_split, test_index_split in kf.split(X):
            subsampled_idxs = rs.choice(len(train_index_split), num_train,replace=False)
            train_idxs_0.append(train_index_split[subsampled_idxs])
            test_idxs_0.append(test_index_split)
            
        train_idxs.append(train_idxs_0)
        test_idxs.append(test_idxs_0)
        
    return train_idxs, test_idxs

def cmn(q, preds):
    sum_f = np.sum(preds)
    cmn_preds = []
    if sum_f > 0 and (len(preds) - sum_f) > 0:
        for f in preds:
            c = f * q / sum_f > (1-q)*(1 - f)/(len(preds) - sum_f)
            cmn_preds.append(c)
    else:
        print("bad case - just returning preds")
        return preds
        #for f in preds:
#            c = 0
            # we should be ignoring this one va
            #cmn_preds.append(int(-1)*np.ones(len(f)))
    return cmn_preds

def define_vectorized_accuracy(y_true, binary,thresh="cmn"):
    '''
    Args:
        y_true (np array): vector of true values with which to compare
        binary (bool): are we doing binary classification?
        thresh (float or "cmn" or None): if we're threshholding on a scalar value, using cmn,
            or none of the above.
    Returns:
        vectorized accuracy function for the specified problem
    Description: defines a vectorized function for computing accuracy.
    '''
    def this_accuracy(y_pred):
        if binary:
            if thresh == "cmn":
                preds_bin = cmn(0.5,y_pred)
            else:
                preds_bin = y_pred > thresh
                
            return(sklearn.metrics.accuracy_score(y_true,preds_bin))
        # not binary
        else:
            return(sklearn.metrics.r2_score(y_true,y_pred))
        
    return np.vectorize(this_accuracy) 

def define_accuracy(y_true, binary,thresh):
    '''
    Args:
        y_true (np array): vector of true values with which to compare
        binary (bool): are we doing binary classification?
        thresh (float or "cmn" or None): if we're threshholding on a scalar value, using cmn,
            or none of the above.
    Returns:
        vectorized accuracy function for the specified problem
    Description: defines a vectorized function for computing accuracy.
    '''
    def this_accuracy(y_pred):
        if binary:
            if thresh == "cmn":
                preds_bin = cmn(0.5,y_pred)
            else:
                preds_bin = y_pred > thresh
                
            return(sklearn.metrics.accuracy_score(y_true,preds_bin))
        # not binary
        else:
            return(sklearn.metrics.r2_score(y_true,y_pred))
        
    return this_accuracy
