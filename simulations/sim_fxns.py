import numpy as np
import sklearn.metrics

import matplotlib.pyplot as plt
import scipy.linalg

# local imports
import sys
sys.path.append("../scripts")
import plot_utils
import unsharpening_matrices as um
import smooth

plot_utils.setup_sim_plots()

def setup_vars_simple(n, noise_scale_X, noise_scale_y, noise_cov_Z, z_mean_function, a=1,random_state=0):
    rs = np.random.RandomState(random_state)
    t = np.arange(n).reshape(-1,1)/n
    
    Z_mean = z_mean_function(t)
    noise_Z = rs.multivariate_normal(np.zeros(n), noise_cov_Z).reshape(-1,1)
    Z = Z_mean + noise_Z
  
    y = a*Z
    X = Z
    
    noise_y = noise_scale_y*rs.randn(y.shape[0],y.shape[1])
    noise_X = noise_scale_X*rs.randn(X.shape[0],X.shape[1])
    y_obs = y + noise_y
    X_obs = X + noise_X
    
    
    return {"X_obs":X_obs, "y_obs":y_obs, 'y_true':y, 
            "Z_mean": Z_mean, "Z": Z, "t":t, "a":a,
           "sigma_x": noise_scale_X, "sigma_y":noise_scale_y, "cov_Z":noise_cov_Z
            }


def setup_vars_simple2d(n, noise_scale_X, noise_scale_y, noise_cov_Z, z_mean_function, c=1):
    rs = np.random.RandomState(0)
    t = np.arange(n).reshape(-1,1)
    
    # input function here
    Z_mean = z_mean_function(t)
    
    Z = np.zeros_like(Z_mean)
    for i in range(Z.shape[1]):
        Z[:,i] = rs.multivariate_normal(Z_mean[:,i],noise_cov_Z)
    
    y = c*Z
    X = Z
    
    noise_y = noise_scale_y*rs.randn(y.shape[0],y.shape[1])
    noise_X = noise_scale_X*rs.randn(X.shape[0],X.shape[1])
    y_obs = y + noise_y
    X_obs = X + noise_X
    
    return {"X_obs":X_obs, "y_obs":y_obs,  "Z_mean": Z_mean, "Z": Z, "t":t}


def split_and_predict(setup_vars, val_pct=0.2,add_intercept=True,random_seed=0, use_tls=False):
    
    X_0, y, Z, t = setup_vars['X_obs'], setup_vars['y_obs'], setup_vars['Z'], setup_vars['t']
    if add_intercept:
        X = np.hstack((X_0, np.ones((X_0.shape[0],1))))
    else:
        X = X_0.reshape(-1,1)

    rs = np.random.RandomState(random_seed)
    assert len(X) == len(y)
    n = len(X)
    n_train = int(n*(1-val_pct))
    n_val = int(n*val_pct)

    random_idxs = rs.choice(n,n,replace=False)
    train_idxs = random_idxs[:n_train]
    val_idxs = random_idxs[n_train:]

    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_val = X[val_idxs]
    y_val = y[val_idxs]
    

    if use_tls:
        w_hat_tls = tls(X_train, y_train)
        w_hat = w_hat_tls.T
    else:
        XTX = X_train.T.dot(X_train)
        XTy = X_train.T.dot(y_train)
        w_hat = np.linalg.solve(XTX,XTy)
    
    y_train_pred = X_train.dot(w_hat)
    y_val_pred = X_val.dot(w_hat)

    return {"X_train":X_train, 
            "X_val":X_val, 
            "y_train":y_train, 
            "y_val":y_val,
            "y_train_pred":y_train_pred,
            "y_val_pred":y_val_pred,
            "w_hat":w_hat, 
            "train_idxs":train_idxs,
            "val_idxs": val_idxs
           }


def make_preds_dict(setup_vars_dict, results_dict, smoothed_dict):
    X_obs, y_obs = setup_vars_dict['X_obs'], setup_vars_dict['y_obs']
    Z, t = setup_vars_dict['Z'], setup_vars_dict['t']

    y_train_pred, y_val_pred = results_dict['y_train_pred'],results_dict['y_val_pred']  
    train_idxs, val_idxs = results_dict['train_idxs'],results_dict['val_idxs']  
       
    preds_dict = {"y_train_pred_smooth":smoothed_dict['y_train_pred_smooth'], 
                  "y_val_pred_smooth": smoothed_dict['y_val_pred_smooth'],
             "y_train_pred_unsmooth":y_train_pred,"y_val_pred_unsmooth":y_val_pred,
             "y_train": y_obs[train_idxs], "y_val":y_obs[val_idxs],
             "Z_train": Z[train_idxs],"Z_val": Z[val_idxs],
             "X_train": X_obs[train_idxs],"X_val": X_obs[val_idxs],
             "t_train": t[train_idxs], "t_val":t[val_idxs]}
    return preds_dict
    
    

def performance_by_sigmas(setup_dict, results_dict, t, sigmas, c1s,c2s,
                          eval_fxn=sklearn.metrics.r2_score):
    '''
    returns will be in the form y_preds_val[s,i,j] = c1*y_val_pred_smooth + c2*y_val_pred
    where i corresponds to c1, j corresponds to c2, s corresponds to simga
    '''
    #y_train, y_val = results_dict['y_train'], results_dict['y_val']
    X_train, X_val = results_dict['X_train'], results_dict['X_val']
    train_idxs, val_idxs = results_dict['train_idxs'], results_dict['val_idxs']
    w_hat = results_dict['w_hat']
    y_train_pred, y_val_pred = np.dot(X_train, w_hat), np.dot(X_val, w_hat)

    y_true = setup_dict['y_obs']
    y_train, y_val = y_true[train_idxs], y_true[val_idxs]
    
    ret_shape = (len(sigmas), len(c1s), len(c2s))
    r2s_train = np.zeros(ret_shape)
    r2s_val = np.zeros(ret_shape)
    y_preds_train = np.zeros(ret_shape+y_train.shape)
    y_preds_val = np.zeros(ret_shape+y_val.shape)                        
    D = um.distance_matrix(t)
    for s,sigma in enumerate(sigmas):
        gamma = um.sigma_to_gamma(sigma)
        W = um.fast_exp(-gamma*D)
        W = W / np.sum(W, axis=1).reshape(-1,1)
        S = W
        

        S_train_train = S[train_idxs,:][:,train_idxs]
        S_val_val = S[val_idxs,:][:,val_idxs]


        y_train_pred_smooth = smooth.smooth(S_train_train,y_train_pred)
        y_val_pred_smooth = smooth.smooth(S_val_val,y_val_pred)
        
        # shrink with c1 and c2
        for i, c1 in enumerate(c1s):
            for j, c2 in enumerate(c2s):
                y_train_pred_smooth_this = c2*(c1*y_train_pred_smooth + (1-c1)*y_train_pred)
                y_val_pred_smooth_this = c2*(c1*y_val_pred_smooth + (1-c1)*y_val_pred)
                y_preds_train[s,i,j] = y_train_pred_smooth_this
                y_preds_val[s,i,j] = y_val_pred_smooth_this

                r2s_train[s,i,j] = eval_fxn(y_train, y_train_pred_smooth_this)
                r2s_val[s,i,j] = eval_fxn(y_val, y_val_pred_smooth_this)

        rets_dict_smoothed = {"r2s_train_smoothed": r2s_train,
                "r2s_val_smoothed": r2s_val,
                "y_preds_train_smoothed": y_preds_train,
                "y_preds_val_smoothed": y_preds_val,
                "sigmas":sigmas,
                "c1s":c1s,
                "c2s":c2s
               }
    return rets_dict_smoothed

def unsmoothed_performance(setup_dict, results_dict, eval_fxn=sklearn.metrics.r2_score):
    X_train, X_val = results_dict['X_train'], results_dict['X_val']
    w_hat = results_dict['w_hat']
    
    train_idxs, val_idxs = results_dict['train_idxs'], results_dict['val_idxs']
    
    y_true = setup_dict['y_obs']
    y_train, y_val = y_true[train_idxs], y_true[val_idxs]
    
    y_train_pred, y_val_pred = np.dot(X_train, w_hat), np.dot(X_val, w_hat)
    
    r2_train = eval_fxn(y_train, y_train_pred)
    r2_val = eval_fxn(y_val, y_val_pred)
    
    return r2_train, r2_val
    
    
def read_smoothed_performance(results_smoothed_dict):
    # unwrap results_smoothe_dict
    sigmas, c1s, c2s = results_smoothed_dict['sigmas'], results_smoothed_dict['c1s'], results_smoothed_dict['c2s']
    
    r2s_train, r2s_val = results_smoothed_dict['r2s_train_smoothed'],results_smoothed_dict['r2s_val_smoothed']
    best_s_idx_train, best_c1_idx_train, best_c2_idx_train = np.unravel_index(np.argmax(r2s_train), r2s_train.shape)
    y_train_pred_smooth = results_smoothed_dict['y_preds_train_smoothed'][best_s_idx_train, best_c1_idx_train, best_c2_idx_train]
    
    best_s_idx_val, best_c1_idx_val, best_c2_idx_val = np.unravel_index(np.argmax(r2s_val), r2s_val.shape)
    y_val_pred_smooth = results_smoothed_dict['y_preds_val_smoothed'][best_s_idx_val, best_c1_idx_val, best_c2_idx_val]
    
    return {'r2s_train': r2s_train[best_s_idx_train],
            'r2s_val': r2s_val[best_s_idx_train],
            'y_train_pred_smooth': y_train_pred_smooth,
            'y_val_pred_smooth': y_val_pred_smooth,
            'best_sigma_train': sigmas[best_s_idx_train],
            'best_sigma_idx_train': best_s_idx_train,
            'best_c1_train': c1s[best_c1_idx_train],
            'best_c1_idx_train': best_c1_idx_train,
            'best_c2_train': c2s[best_c2_idx_train],
            'best_c2_idx_val': best_c2_idx_val,
            'best_sigma_val': sigmas[best_s_idx_val],
            'best_sigma_idx_val': best_s_idx_val,
            'best_c1_val': c1s[best_c1_idx_val],
            'best_c1_idx_val': best_c1_idx_val,
            'best_c2_val': c2s[best_c2_idx_val],
            'best_c2_idx_val': best_c2_idx_val
    }
    

def read_mse_smoothed(preds_dict):
    y_val = preds_dict["y_val"]
    y_val_pred_smooth = preds_dict["y_val_pred_smooth"]
    return sklearn.metrics.mean_squared_error(y_val,y_val_pred_smooth)
       
def read_mse_unsmoothed(preds_dict):
    y_val = preds_dict["y_val"]
    y_val_pred_unsmooth = preds_dict["y_val_pred_unsmooth"]
    return sklearn.metrics.mean_squared_error(y_val,y_val_pred_unsmooth)
                                              
    
def plot_fancy_scatter(preds_dict,vars_dict, figsize=(36,6), s = 1, fig=None, ax=None,title_columns=True,cmap=None,
                      predictor="o.l.s."):
    # plot three column of scatter plots showing: local structure, unsmoothed preds, and smoothed preds
    y_train_pred_unsmooth, y_val_pred_unsmooth = preds_dict["y_train_pred_unsmooth"], preds_dict["y_val_pred_unsmooth"]
    y_train_pred_smooth, y_val_pred_smooth = preds_dict["y_train_pred_smooth"], preds_dict["y_val_pred_smooth"]
    y_train, y_val = preds_dict["y_train"], preds_dict["y_val"]
    Z_train, Z_val = preds_dict["Z_train"], preds_dict["Z_val"]
    X_train, X_val = preds_dict["X_train"], preds_dict["X_val"]
    t_train, t_val = preds_dict['t_train'], preds_dict['t_val']
    noise_scale_X, noise_scale_y = vars_dict['noise_scale_x'], vars_dict['noise_scale_y']
      
    if ax is None:
        fig, ax = plt.subplots(1,3, figsize=figsize)#, sharey=True)
    else:
        assert len(ax)>=3
        assert not fig is None
    ax[0].scatter(t_val,y_val, c=t_val,s=s,edgecolors=None,cmap=cmap)
    ax[0].set_xlabel("location $t$")
    ax[0].set_ylabel("observed label $y(t)$")
    
    ax[1].scatter(y_val_pred_unsmooth,y_val, c=t_val,s=s,edgecolors=None,cmap=cmap)
    ax[1].set_xlabel("prediction $\widehat{y}(t)$ ")
    ax[1].set_ylabel("observed label $y(t)$")
    
    ax2 = ax[2].scatter(y_val_pred_smooth,y_val, c=t_val,s=s,edgecolors=None,cmap=cmap)

    ax[2].set_xlabel("smoothed prediction $\widetilde{y}(t)$")
    ax[2].set_ylabel("observed label $y(t)$")
    
    if title_columns:
        ax[0].set_title("underlying neighborhood structure \n $\sigma_x = {0}$, $\sigma_y = {1}$".format(
            noise_scale_X, noise_scale_y))
        ax[1].set_title("{1} predictions \n $R^2$ = {0:.02}".format(
            sklearn.metrics.r2_score(y_val,y_val_pred_unsmooth), predictor))
        ax[2].set_title("smoothed predictions \n $R^2$ = {0:.02} ".format(
            sklearn.metrics.r2_score(y_val,y_val_pred_smooth)))
    else:
        ax[0].set_title("$\sigma_x = {0}$, $\sigma_y = {1}$".format(
            noise_scale_X, noise_scale_y)) 
        ax[1].set_title("$R^2$ = {0:.02}".format(
            sklearn.metrics.r2_score(y_val,y_val_pred_unsmooth)))
        ax[2].set_title("$R^2$ = {0:.02} ".format(
            sklearn.metrics.r2_score(y_val,y_val_pred_smooth)))
    
    cbaxes = fig.add_axes([.92, 0.1, 0.01, 0.8]) 
    cb = plt.colorbar(ax2, cax = cbaxes)  
    cb.set_ticks([0.0,0.5,1.0])
    cb.set_ticklabels([0.0,0.5,1.0])
    cbaxes.set_title('t')
    
    return sklearn.metrics.mean_squared_error(y_val,y_val_pred_smooth)

def smooth_with_W_star(setup_vars_dict, results_dict):
    W_star = get_W_star(setup_vars_dict,results_dict, compute_alpha=True)
    smoothed = W_star.dot(results_dict['y_val_pred'])
    return smoothed



def get_W_star(setup_vars_dict, results_dict, compute_alpha = True, sort=False):
    if sort:
        val_idxs = np.sort(results_dict['val_idxs'])
    else:
        val_idxs = results_dict['val_idxs']
    K_zz = setup_vars_dict['cov_Z'][:,val_idxs][val_idxs]
    a  = setup_vars_dict["a"]
    K_yyhat = a**2 * K_zz
    
    sigma_w = setup_vars_dict['sigma_x']
    sigma_mu = setup_vars_dict['sigma_y']
    sigma_gamma_sq = sigma_w**2 + a**2 * sigma_mu**2
    
    if compute_alpha:
        n_train = results_dict['X_train'].shape[0]
        emp_cov_X = np.cov(results_dict['X_train'].ravel())
        alpha = sigma_gamma_sq / (n_train-1) * emp_cov_X 
    else:
        alpha = 0
    K_yhatyhat = (a**2+alpha)*(K_zz + sigma_w**2 * np.eye(K_zz.shape[0]))
    W_star = scipy.linalg.solve(K_yhatyhat, K_yyhat,sym_pos=True).T
    
    return W_star

def compute_expectation_unsmoothed(setup_vars_dict, results_dict,compute_alpha = True):
    a = setup_vars_dict['a']
    
    val_idxs = results_dict['val_idxs']
    val_idxs_sorted = np.sort(val_idxs)
    n_val = len(val_idxs)
    I = np.eye(n_val)
    
    K_zz = setup_vars_dict['cov_Z'][:,val_idxs_sorted][val_idxs_sorted]
    K_yy = a**2 * K_zz + setup_vars_dict['sigma_y']**2 * I 
    K_yyhat = a**2 * K_zz
    K_yhatyhat = a**2 * (K_zz + setup_vars_dict['sigma_x']**2 * I )
    K_ee = K_yhatyhat - 2 *K_yyhat + K_yy
    return 1/n_val * np.trace(K_ee)

def compute_expectation_lower_bound(setup_vars_dict, results_dict, val_only=True, compute_alpha = True):
    if val_only:
        val_idxs = results_dict['val_idxs']
        K_zz = setup_vars_dict['cov_Z'][:,val_idxs][val_idxs]
        n = len(val_idxs)
        
    else: 
        K_zz = setup_vars_dict['cov_Z']
        n = K_zz.shape[0]
        
    a = setup_vars_dict['a']
    
    sigma_w = setup_vars_dict['sigma_x']
    sigma_mu = setup_vars_dict['sigma_y']
    
    K_ww = sigma_w**2 * np.eye(K_zz.shape[0])
    K_mumu = sigma_mu**2 * np.eye(K_zz.shape[0])
    
    K_yy = a**2 * K_zz + K_mumu
    K_yyhat = a**2 * K_zz

    if compute_alpha:
        sigma_gamma_sq = sigma_w**2 + a**2 * sigma_mu**2
        n_train = results_dict['X_train'].shape[0]
        emp_cov_X = np.cov(results_dict['X_train'].ravel())
        alpha = sigma_gamma_sq / (n_train-1) * emp_cov_X 
    else:
        alpha = 0
    K_yhatyhat = (a**2+alpha)*(K_zz + sigma_w**2 * np.eye(K_zz.shape[0]))
    K_yhatyhat_inv = scipy.linalg.solve(K_yhatyhat, np.eye(K_yhatyhat.shape[0]),sym_pos=True)
    
    return 1/n * np.trace(K_yy - K_yyhat.dot(K_yhatyhat_inv).dot(K_yyhat))


# total least squares implementation
def tls(X,y):
    (n,d) = X.shape
    Z = np.hstack((X,y))
    U,S,Vh = np.linalg.svd(Z, full_matrices=True)
    V = Vh.T
    V_xy = V[:d, d:]
    V_yy = V[d:, d:]
    B = - V_xy/V_yy
    
    return B
    