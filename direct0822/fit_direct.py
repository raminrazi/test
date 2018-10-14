from scipy.optimize import minimize
import sys
sys.path.append('./ODEs/')
from fitzhugh import *
from rossler import *
from lotka_volterra import *
from lorenz96 import *

def fit_direct(Y,dt,init_params,ODE_str,lam = 1,max_iters=10000, tol=1e-8):
    '''
    This function learns the ODE parameters given the noisy observations.

    Input:
        Y: T*d numpy array that contains the noisy observations.
        dt: Time interval between the observations (states).
        init_params: initialization for the unknown parameters.
        ODE_str: This is a string, with the name of ODE model.
            ODE_str can be set to 'fitzhugh' or 'lotka_volterra' or 'rossler'.
        lam: The hyper-parameter lambda in our method.
        max_iters: Maximum number of iterations
        tol: tolerance value to stop the optimization, if the amount of changes in the objective is small.
    Output:
        params: The estimated parameters.
        X: The estimated states.
        pred_X: The predicted states.
    '''

    # Based on the name of the ODE in ODE_str, we set the name of the functions
    param_obj_fun = eval(ODE_str + '_param_obj')  # Function to evaluate the obj in eq. (8) of the paper.
    param_grad_fun = eval(ODE_str + '_param_grad')  # Function that returns gradient of the obj in eq. (8) wrt. parameters.
    X_obj_fun = eval(ODE_str + '_X_obj')  # Function to evaluate the obj in eq. (9) of the paper.
    X_grad_fun = eval(ODE_str + '_X_grad')  # Function that returns gradient of the obj of eq. (9) wrt. parameters
    pred_fun = eval(ODE_str + '_predict')  # Function that predicts states by repeatedly applying eq. (6) of the paper.

    # Initialization of states and parameters
    X = Y
    params = init_params

    T, d = X.shape
    new_cost = 1000

    #main loop
    for k in range(max_iters):

        # optimization over parameters given states
        #num1 = param_obj_fun(params,X,dt)
        #num2 = param_grad_fun(params,X,dt)

        res = minimize(param_obj_fun, params, method='L-BFGS-B', jac=param_grad_fun, args=(X, dt),
               options={'disp': False,'maxcor': 100})
        params = res['x']

        # stop if the changes in the objective is smaller than tol
        prev_cost = new_cost
        new_cost = param_obj_fun(params, X, dt)
        if((prev_cost - new_cost) < tol and k > 1):
            break

        if (k % 1000 == 0):
            print('iter', k, 'params:', params, 'obj_val:', new_cost)

        # optimization over the states given the parameters
        X0 = X.flatten('F')

        res = minimize(X_obj_fun, X0, method='L-BFGS-B', jac=X_grad_fun, args=(params, dt,X0,lam,d),
                           options={'disp': False,'maxcor': 100})
        x = res['x']

        X = x.reshape((d,T)).T
        #X = x.reshape(T, -1)
    # Predict the states by applying eq. (6) of the paper
    pred_X = pred_fun(X[0,:],T,dt,params)

    return params,X,pred_X



