import numpy as np


def fitzhugh_ode(t, x, params):
    '''
    This is the ODE function f(x,params). This function will be sent as a parameter to the scipy.integrate.ode().
    To see how this works, please look at the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
        x: 2-dimensional state at time t.
        params: 3-dimensional parameters.

    Output:
        2-dimensional derivative dx/dt=[x0_dot, x1_dot].
    '''

    a, b, c = params[0], params[1], params[2]
    x0_dot = c*(x[0]-((x[0]**3)/3)+x[1])
    x1_dot = -(1/c)*(x[0] -a+ b*x[1])
    return [x0_dot,x1_dot]


def fitzhugh_predict(init_state, T, dt, params):
    '''
    We use this function to predict the states, by applying eq. (6) of the paper repeatedly.

    Input:
        init_state: the initial 2-dimensional state.
        T: Total number of states to predict.
        dt: Time interval between the states.
        params: 3-dimensional parameters.

    Output:
        pX: The predicted states, a T*2 numpy array
    '''
    a,b,c = params[0],params[1],params[2]
    pX = np.zeros((T,2))
    pX[0,:] = init_state
    for i in range(1, T):
        pX[i,0] = pX[i - 1,0] + c*(pX[i - 1,0] - (pX[i - 1,0]**3)/3 + pX[i - 1,1])*dt
        pX[i, 1] = pX[i - 1, 1] + (-1/c)*(pX[i-1,0] - a + (b * pX[i-1, 1]))*dt
    return pX


def fitzhugh_X_obj(x, params, dt, x0, lam, d):
    '''
    This function returns the objective value in eq. (9) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        x: A 1 by (T*2) numpy vector. This is the flattened version of the states X, which is T by 2.
        params: 3-dimensional parameters.
        dt: Time interval between the observations (states).
        x0: Initialization of the states for the optimization.
        lam: The hyperparameter lambda in our paper.

    Output:
        objval: The objective value
    '''
    a,b,c = params[0],params[1],params[2]
    T = x.shape[0]
    T = int(T / 2)
    x1 = x[0:T]
    x2 = x[T:2*T]
    t1 = x1[1:T] - x1[0:T-1] - (c*dt*(x1[0:T-1] - ((x1[0:T-1]**3)/3) + x2[0:T-1]))
    t2 = x2[1:T] - x2[0:T-1] + (1/c)*(x1[0:T-1] -a + (b*x2[0:T-1]))*dt
    objval = (np.linalg.norm(t1)**2) + (np.linalg.norm(t2)**2) + lam*(np.linalg.norm(x-x0)**2)
    return objval


def fitzhugh_X_grad(x, params, dt, x0, lam ,d):
    '''
    This function returns the gradient with respect to the states X in the eq. (9) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        Please look at the previous function 'fitzhugh_X_obj()' for the input variables.

    Output:
        g: It has the same size as the input x and contains the gradients.
    '''
    a,b,c = params[0],params[1],params[2]
    T = x.shape[0]
    T = int(T / 2)
    x1 = x[0:T]
    x2 = x[T:2*T]
    x01 = x0[0:T]
    x02 = x0[T:2*T]
    term1 = np.zeros_like(x1)
    term2 = np.zeros_like(x1)
    term3 = np.zeros_like(x1)
    term4 = np.zeros_like(x1)
    term1[0:T-1] = x1[1:T] - x1[0:T-1] - (c*dt*(x1[0:T-1] - ((x1[0:T-1]**3)/3) + x2[0:T-1]))
    term2[1:T] = term1[0:T-1]
    term3[0:T-1] = x2[1:T] - x2[0:T-1] + (1/c)*(x1[0:T-1] -a + (b*x2[0:T-1]))*dt
    term4[1:T] = term3[0:T-1]
    g1 = (2 * term1 * (-1 - c*dt*(1 - ((x1**2))))) + (2 * term2) + 2 * term3 * (1/c) * dt
    g2 = 2*term1*((-c)*dt) + 2*term3*(-1 + (b*dt*(1/c))) + 2*term4
    g = np.zeros_like(x)
    g1 = g1 + 2*lam*(x1 - x01)
    g2 = g2 + 2*lam*(x2-x02)
    g[0:T] = g1
    g[T:2*T] = g2
    return g


def fitzhugh_param_obj(params, X, dt):
    '''
    This function returns the objective value in eq. (8) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        params: 3-dimensional parameters.
        X: T*2 numpy vector, which contains the states.
        dt: Time interval between the state observations.
    Output:
        objval: The objective value
    '''

    T = X.shape[0]
    term1 = X[1:T,0] - X[0:T-1,0] - params[2]*(X[0:T-1,0]- ((X[0:T-1,0]**3)/3) + X[0:T-1,1])*dt
    term1 = np.sum(term1**2)
    term2 = X[1:T,1] - X[0:T-1,1] + (1/params[2])*(X[0:T-1,0] - params[0] + params[1]*X[0:T-1,1])*dt
    term2 = np.sum(term2**2)
    objval = term1 + term2
    return objval


def fitzhugh_param_grad(params, X, dt):
    '''
    This function returns the gradient with respect to the parameters in the eq. (8) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        please look at the previous function 'fitzhugh_param_obj()' for the input variables.

    Output:
       g: It has the same size as the params and contains the gradients.
    '''

    T = X.shape[0]
    term1 = X[1:T, 0] - X[0:T - 1, 0] - params[2] * (X[0:T - 1, 0] - ((X[0:T - 1, 0] ** 3) / 3) + X[0:T - 1, 1]) * dt
    term2 = X[1:T, 1] - X[0:T - 1, 1] + (1 / params[2]) * (X[0:T - 1, 0] - params[0] + params[1] * X[0:T - 1, 1]) * dt
    term1c = 2*term1*-1* (X[0:T - 1, 0] - ((X[0:T - 1, 0] ** 3) / 3) + X[0:T - 1, 1]) * dt
    term2a = 2*term2*(-1/params[2])*dt
    term2b = 2*term2*(1/params[2])*X[0:T-1, 1]*dt
    term2c = 2*term2*((-1/(params[2]**2)))* (X[0:T - 1, 0] - params[0] + params[1] * X[0:T - 1, 1]) * dt
    ga = np.sum(term2a)
    gb = np.sum(term2b)
    gc = np.sum(term1c+term2c)
    g = np.array([ga,gb,gc])
    return g