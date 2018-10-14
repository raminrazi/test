import numpy as np


def rossler_ode(t, x, params):
    '''
    This is the ODE function f(x,params). This function will be sent as a parameter to the scipy.integrate.ode().
    To see how this works, please look at the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
        x: 3-dimensional state at time t.
        params: 3-dimensional parameters.

    Output:
        3-dimensional derivative dx/dt=[x0_dot, x1_dot, x2_dot].
    '''

    a, b, c = params[0], params[1], params[2]

    x0_dot = -x[1] - x[2]
    x1_dot = x[0] + a*x[1]
    x2_dot = b + (x[2] * (x[0] - c) )
    return [x0_dot,x1_dot,x2_dot]

def rossler_predict(init_state, T, dt, params):
    '''
    We use this function to predict the states, by applying eq. (6) of the paper repeatedly.

    Input:
        init_state: the initial 3-dimensional state.
        T: Total number of states to predict.
        dt: Time interval between the states.
        params: 3-dimensional parameters.

    Output:
        pX: The predicted states, a T*3 numpy array
    '''
    a, b, c = params[0], params[1], params[2]

    pX = np.zeros((T, 3))
    pX[0, :] = init_state
    for i in range(1, T):
        pX[i, 0] = pX[i - 1, 0] + (-pX[i - 1, 1] - pX[i - 1, 2]) * dt
        pX[i, 1] = pX[i - 1, 1] + (pX[i - 1, 0] + (a * pX[i - 1, 1])) * dt
        pX[i, 2] = pX[i - 1, 2] + (b + (pX[i - 1, 2] * (pX[i - 1, 0] - c))) * dt
    return pX


def rossler_X_obj(x, params, dt, x0, lam, d):
    '''
    This function returns the objective value in eq. (9) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        x: A 1 by (T*3) numpy vector. This is the flattened version of the states X, which is T by 3.
        params: 3-dimensional parameters.
        dt: Time interval between the observations (states).
        x0: Initialization of the states for the optimization.
        lam: The hyperparameter lambda in our paper.

    Output:
        objval: The objective value
    '''
    a, b, c = params[0], params[1], params[2]
    T = x.shape[0]
    T = int(T / 3)
    ox = np.copy(x)
    x = ox[0:T]
    y = ox[T:2 * T]
    z = ox[2 * T:3 * T]
    t1 = x[1:T] - x[0:T - 1] - (-y[0:T - 1] - z[0:T - 1]) * dt
    t2 = y[1:T] - y[0:T - 1] - (x[0:T - 1] + (a * y[0:T - 1])) * dt
    t3 = z[1:T] - z[0:T - 1] - (b + (z[0:T - 1] * (x[0:T - 1] - c))) * dt
    objval = (np.linalg.norm(t1) ** 2) + (np.linalg.norm(t2) ** 2) + \
          (np.linalg.norm(t3) ** 2) + lam * (np.linalg.norm(ox - x0) ** 2)
    return objval


def rossler_X_grad(x, params, dt, x0, lam, d):
    '''
    This function returns the gradient with respect to the states X in the eq. (9) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        Please look at the previous function 'rossler_X_obj()' for the input variables.

    Output:
        g: It has the same size as the input x and contains the gradients.
    '''
    a, b, c = params[0], params[1], params[2]
    T = x.shape[0]
    T = int(T / 3)
    ox = np.copy(x)
    x = ox[0:T]
    y = ox[T:2 * T]
    z = ox[2 * T:3 * T]
    x01 = x0[0:T]
    x02 = x0[T:2 * T]
    x03 = x0[2 * T:3 * T]
    term1 = np.zeros_like(x)
    term2 = np.zeros_like(x)
    term3 = np.zeros_like(x)
    term4 = np.zeros_like(x)
    term5 = np.zeros_like(x)
    term6 = np.zeros_like(x)
    term1[0:T - 1] = x[1:T] - x[0:T - 1] - (-y[0:T - 1] - z[0:T - 1]) * dt
    term2[1:T] = term1[0:T - 1]
    term3[0:T - 1] = y[1:T] - y[0:T - 1] - (x[0:T - 1] + (a * y[0:T - 1])) * dt
    term4[1:T] = term3[0:T - 1]
    term5[0:T - 1] = z[1:T] - z[0:T - 1] - (b + (z[0:T - 1] * (x[0:T - 1] - c))) * dt
    term6[1:T] = term5[0:T - 1]
    g1 = 2 * term1 * (-1) + 2 * term2 + 2 * term3 * (-dt) + 2 * term5 * (-z * dt)
    g2 = 2 * term1 * (dt) + 2 * term3 * (-1 - (a * dt)) + 2 * term4
    g3 = 2 * term1 * (dt) + 2 * term5 * (-1 - ((x - c) * dt)) + 2 * term6
    g = np.zeros_like(ox)
    g[0:T] = g1 + 2 * lam * (x - x01)
    g[T:2 * T] = g2 + 2 * lam * (y - x02)
    g[2 * T:3 * T] = g3 + 2 * lam * (z - x03)
    return g


def rossler_param_obj(params, X, dt):
    '''
    This function returns the objective value in eq. (8) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        params: 3-dimensional parameters.
        X: T*3 numpy vector, which contains the states.
        dt: Time interval between the state observations.
    Output:
        objval: The objective value
    '''

    T = X.shape[0]
    a,b,c = params[0],params[1],params[2]
    temp1 = X[1:T, 0] - X[0:T - 1, 0] - ((-X[0:T - 1, 1] - X[0:T - 1, 2]) * dt)
    term1 = np.sum((temp1) ** 2)
    temp1 = X[1:T, 1] - X[0:T - 1, 1] - ((X[0:T - 1, 0] + (a * X[0:T - 1, 1])) * dt)
    term2 = np.sum((temp1) ** 2)
    temp1 = X[1:T, 2] - X[0:T - 1, 2] - ((b + (X[0:T - 1, 2] * (X[0:T - 1, 0] - c))) * dt)
    term3 = np.sum((temp1) ** 2)
    objval = term1 + term2 + term3
    return objval


def rossler_param_grad(params, X, dt):
    '''
    This function returns the gradient with respect to the parameters in the eq. (8) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        please look at the previous function 'rossler_param_obj()' for the input variables.

    Output:
       g: It has the same size as the params and contains the gradients.
    '''
    a, b, c = params[0], params[1], params[2]
    T = X.shape[0]
    term1 = (X[1:T, 1] - X[0:T - 1, 1] - ((X[0:T - 1, 0] + (a * X[0:T - 1, 1])) * dt)) * (-X[0:T - 1, 1]) * dt
    term2 = (X[1:T, 2] - X[0:T - 1, 2] - ((b + X[0:T - 1, 2] * (X[0:T - 1, 0] - c)) * dt)) * (-dt)
    term3 = (X[1:T, 2] - X[0:T - 1, 2] - ((b + X[0:T - 1, 2] * (X[0:T - 1, 0] - c)) * dt)) * (X[0:T - 1, 2]) * (dt)
    ga = np.sum(2 * term1)
    gb = np.sum(2 * term2)
    gc = np.sum(2 * term3)
    g = np.array([ga, gb, gc])
    return g




