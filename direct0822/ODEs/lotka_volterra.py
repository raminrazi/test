import numpy as np


def lotka_volterra_ode(t,x, params):
    '''
    This is the ODE function f(x,params). This function will be sent as a parameter to the scipy.integrate.ode().
    To see how this works, please look at the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
        x: 2-dimensional state at time t.
        params: 4-dimensional parameters.

    Output:
        2-dimensional derivative dx/dt=[x0_dot, x1_dot].
    '''
    a, b, c, d = params[0], params[1], params[2], params[3]

    x0_dot = a*x[0] - b*x[0]*x[1]
    x1_dot = -c*x[1] + d*x[0]*x[1]
    return [x0_dot,x1_dot]


def lotka_volterra_predict(init_state, T, dt, params):
    a, b, c, d = params[0], params[1], params[2], params[3]
    pX = np.zeros((T,2))
    pX[0,:] = init_state
    for i in range(1, T):
        pX[i, 0] = pX[i - 1, 0] + (a * pX[i - 1, 0] - b * pX[i - 1, 0] * pX[i - 1, 1]) * dt
        pX[i, 1] = pX[i - 1, 1] + (-c * pX[i - 1, 1] + d * pX[i - 1, 0] * pX[i - 1, 1]) * dt
    return pX


def lotka_volterra_X_obj(x, params, dt, x0, lam, d):
    '''
    This function returns the objective value in eq. (9) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        x: A 1 by (T*2) numpy vector. This is the flattened version of the states X, which is T by 2.
        params: 4-dimensional parameters.
        dt: Time interval between the observations (states).
        x0: Initialization of the states for the optimization.
        lam: The hyperparameter lambda in our paper.

    Output:
        objval: The objective value
    '''

    a, b, c, d = params[0], params[1], params[2], params[3]
    T = x.shape[0]
    T = int(T / 2)
    x1 = x[0:T]
    x2 = x[T:2*T]
    term1 = x1[1:T] - x1[0:T - 1] - (a * x1[0:T - 1] - b * x1[0:T - 1] * x2[0:T - 1]) * dt
    term2 = x2[1:T] - x2[0:T - 1] - (-c * x2[0:T - 1] + d * x1[0:T - 1] * x2[0:T - 1]) * dt
    objval = (np.linalg.norm(term1)**2) + (np.linalg.norm(term2)**2) + 2*lam*(np.linalg.norm(x-x0)**2)
    return objval


def lotka_volterra_X_grad(x, params, dt, x0, lam, d):
    '''
    This function returns the gradient with respect to the states X in the eq. (9) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        Please look at the previous function 'lotka_volterra_X_obj()' for the input variables.

    Output:
        g: It has the same size as the input x and contains the gradients.
    '''

    a,b,c,d = params[0],params[1],params[2],params[3]
    T = x.shape[0]
    T = int(T / 2)
    x1 = x[0:T]
    x2 = x[T:2*T]
    x01 = x0[0:T]
    x02 = x0[T:2 * T]
    term1 = np.zeros_like(x1)
    term2 = np.zeros_like(x1)
    term3 = np.zeros_like(x1)
    term4 = np.zeros_like(x1)
    term1[0:T-1] = x1[1:T] - x1[0:T - 1] - (a * x1[0:T - 1] - b * x1[0:T - 1] * x2[0:T - 1]) * dt
    term2[1:T] = term1[0:T-1]
    term3[0:T-1] = x2[1:T] - x2[0:T - 1] - (-c * x2[0:T - 1] + d * x1[0:T - 1] * x2[0:T - 1]) * dt
    term4[1:T] = term3[0:T-1]
    g1 = 2 * term1 * (-1 - a*dt + (b * x2[0:T]*dt)) + 2 * term2 + 2 * term3 * (-d * x2[0:T])*dt
    g2 = 2 * term1 * (b * x1[0:T] * dt) + 2 * term3 * (-1 + c * dt - (d * x1[0:T ] * dt)) + 2 * term4
    g = term1 = np.zeros_like(x)
    g1 = g1 + 2 * lam * (x1 - x01)
    g2 = g2 + 2*lam*(x2-x02)
    g[0:T] = g1
    g[T:2*T] = g2
    return g


def lotka_volterra_param_obj(params, X, dt):
    '''
    This function returns the objective value in eq. (8) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        params: 4-dimensional parameters.
        X: T*2 numpy vector, which contains the states.
        dt: Time interval between the state observations.
    Output:
        objval: The objective value
    '''

    a, b, c, d = params[0], params[1], params[2], params[3]
    T = X.shape[0]
    term1 = X[1:T,0] - X[0:T - 1,0] - (a * X[0:T - 1,0] - b * X[0:T - 1,0] * X[0:T - 1,1]) * dt
    term2 = X[1:T,1] - X[0:T - 1,1] - (-c * X[0:T - 1,1] + d * X[0:T - 1,0] * X[0:T - 1,1]) * dt
    objval = (np.linalg.norm(term1) ** 2) + (np.linalg.norm(term2) ** 2)
    return objval


def lotka_volterra_param_grad(params, X, dt):
    '''
    This function returns the gradient with respect to the parameters in the eq. (8) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        please look at the previous function 'lotka_volterra_param_obj()' for the input variables.

    Output:
       g: It has the same size as the params and contains the gradients.
    '''
    a, b, c, d = params[0], params[1], params[2], params[3]
    T = X.shape[0]
    term1 = X[1:T, 0] - X[0:T - 1, 0] - (a * X[0:T - 1, 0] - b * X[0:T - 1, 0] * X[0:T - 1, 1]) * dt
    term2 = X[1:T, 1] - X[0:T - 1, 1] - (-c * X[0:T - 1, 1] + d * X[0:T - 1, 0] * X[0:T - 1, 1]) * dt

    ga = 2 * term1 * -1 * X[0:T - 1, 0] * dt
    gb = 2 * term1 * X[0:T - 1, 0] * X[0:T - 1, 1] * dt
    gc = 2 * term2 * X[0:T - 1, 1] * dt
    gd = 2 * term2 * -1 * X[0:T - 1, 0] * X[0:T - 1, 1] * dt
    ga = np.sum(ga)
    gb = np.sum(gb)
    gc = np.sum(gc)
    gd = np.sum(gd)
    g = np.array([ga,gb,gc,gd])
    return g