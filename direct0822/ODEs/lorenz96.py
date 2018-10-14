import numpy as np

def lorenz96_ode(t, x, params):
    '''
    This is the ODE function f(x,params). This function will be sent as a parameter to the scipy.integrate.ode().
    To see how this works, please look at the simulate.py file.

    Input:
        t: time. This should always be the first parameter.
        x: d-dimensional state at time t.
        params: 1-dimensional parameter.

    Output:
        d-dimensional derivative dx/dt=[x0_dot, x1_dot,...].
    '''
    F = params[0]
    T = len(x)
    d = np.zeros((T, 1))
    d[0] = ((x[1] - x[T - 2]) * x[T - 1]) - x[0]
    d[1] = ((x[2] - x[T - 1]) * x[0]) - x[1]
    d[T - 1] = ((x[0] - x[T - 3]) * x[T - 2]) - x[T - 1]
    for i in range(2, T - 1):
        d[i] = ((x[i + 1] - x[i - 2]) * x[i - 1]) - x[i]
    d = d + F
    ld = d.tolist()
    return ld

def lorenz96_predict(init_state, T, dt, params):
    '''
    We use this function to predict the states, by applying eq. (6) of the paper repeatedly.

    Input:
        init_state: the initial d-dimensional state.
        T: Total number of states to predict.
        dt: Time interval between the states.
        params: 1-dimensional parameter.

    Output:
        pX: The predicted states, a T*d numpy array
    '''
    F = params[0]
    L = init_state.shape[0]
    pX = np.zeros((T,L))
    pX[0,:] = init_state
    for i in range(1, T):
        pX[i,0] = pX[i - 1,0] + ((pX[i-1,1] - pX[i-1,L-2])*pX[i-1,L-1] - pX[i-1,0]+F)*dt
        pX[i, 1] = pX[i - 1, 1] + ((pX[i-1,2] - pX[i-1,L-1])*pX[i-1,0] - pX[i-1,1]+F)*dt
        pX[i, L-1] = pX[i - 1, L-1] + ((pX[i-1,0] - pX[i-1,L-3])*pX[i-1,L-2] - pX[i-1,L-1]+F)*dt
        for k in range(2,L-1):
            pX[i, k] = pX[i - 1, k] + ((pX[i-1,k+1] - pX[i-1,k-2])*pX[i-1,k-1] - pX[i-1,k]+F)*dt
    return pX


def lorenz96_X_obj(x, params, dt, x0, lam, d):
    '''
    This function returns the objective value in eq. (9) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        x: A 1 by (T*d) numpy vector. This is the flattened version of the states X, which is T by d.
        params: 1-dimensional parameter.
        dt: Time interval between the observations (states).
        x0: Initialization of the states for the optimization.
        lam: The hyperparameter lambda in our paper.
        d: dimension of the states

    Output:
        objval: The objective value
    '''

    F=params[0]
    T = x.shape[0]
    T = int(T / d)
    #X = x.reshape(T, -1)
    X = x.reshape((d,T)).T

    terms = np.zeros((T - 1, d))
    temp1 = ((X[0:T - 1, 3:d] - X[0:T - 1, 0:d - 3]) * X[0:T - 1, 1:d - 2] - X[0:T - 1, 2:d - 1] + F) * dt
    terms[:, 2:d - 1] = X[1:T, 2:d - 1] - X[0:T - 1, 2:d - 1] - temp1

    temp1 = ((X[0:T - 1, 1] - X[0:T - 1, d - 2]) * X[0:T - 1, d - 1] - X[0:T - 1, 0] + F) * dt
    terms[:, 0] = X[1:T, 0] - X[0:T - 1, 0] - temp1

    temp1 = ((X[0:T - 1, 2] - X[0:T - 1, d - 1]) * X[0:T - 1, 0] - X[0:T - 1, 1] + F) * dt
    terms[:, 1] = X[1:T, 1] - X[0:T - 1, 1] - temp1

    temp1 = ((X[0:T - 1, 0] - X[0:T - 1, d - 3]) * X[0:T - 1, d - 2] - X[0:T - 1, d - 1] + F) * dt
    terms[:, d - 1] = X[1:T, d - 1] - X[0:T - 1, d - 1] - temp1

    res = np.sum((terms) ** 2)
    objval = res + lam * (np.linalg.norm(x - x0) ** 2)
    return objval


def lorenz96_X_grad(x, params, dt, x0, lam, d):
    '''
    This function returns the gradient with respect to the states X in the eq. (9) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        Please look at the previous function 'lorenz96_X_obj()' for the input variables.

    Output:
        g: It has the same size as the input x and contains the gradients.
    '''
    
    F = params[0]
    T = x.shape[0]
    T = int(T / d)
    X = x.reshape((d,T)).T
    X0 = x0.reshape((d,T)).T
    #X = x.reshape(T, -1)
    #X0 = x0.reshape(T, -1)

    terms = np.zeros((T - 1, d))
    temp1 = ((X[0:T - 1, 3:d] - X[0:T - 1, 0:d - 3]) * X[0:T - 1, 1:d - 2] - X[0:T - 1, 2:d - 1] + F) * dt
    terms[:, 2:d - 1] = X[1:T, 2:d - 1] - X[0:T - 1, 2:d - 1] - temp1

    temp1 = ((X[0:T - 1, 1] - X[0:T - 1, d - 2]) * X[0:T - 1, d - 1] - X[0:T - 1, 0] + F) * dt
    terms[:, 0] = X[1:T, 0] - X[0:T - 1, 0] - temp1

    temp1 = ((X[0:T - 1, 2] - X[0:T - 1, d - 1]) * X[0:T - 1, 0] - X[0:T - 1, 1] + F) * dt
    terms[:, 1] = X[1:T, 1] - X[0:T - 1, 1] - temp1

    temp1 = ((X[0:T - 1, 0] - X[0:T - 1, d - 3]) * X[0:T - 1, d - 2] - X[0:T - 1, d - 1] + F) * dt
    terms[:, d - 1] = X[1:T, d - 1] - X[0:T - 1, d - 1] - temp1

    dervs = np.zeros_like(X)
    dervs[1:T - 1, 2:d - 2] = (2 * terms[1:T - 1, 1:d - 3] * (-1 * X[1:T - 1, 0:d - 4] * dt)) + \
                              (2 * terms[1:T - 1, 2:d - 2] * (-1 + dt)) + \
                              (2 * terms[1:T - 1, 3:d - 1] * (-1) * (X[1:T - 1, 4:d] - X[1:T - 1, 1:d - 3]) * dt) + \
                              (2 * terms[1:T - 1, 4:d] * X[1:T - 1, 3:d - 1] * dt) + (2 * terms[0:T - 2, 2:d - 2])

    dervs[0, 2:d - 2] = (2 * terms[0, 1:d - 3] * (-1 * X[0, 0:d - 4] * dt)) + \
                        (2 * terms[0, 2:d - 2] * (-1 + dt)) + \
                        (2 * terms[0, 3:d - 1] * (-1) * (X[0, 4:d] - X[0, 1:d - 3]) * dt) + \
                        (2 * terms[0, 4:d] * X[0, 3:d - 1] * dt)

    dervs[T - 1, 2:d - 2] = (2 * terms[T - 2, 2:d - 2])

    # dimension 0

    dervs[1:T - 1, 0] = (2 * terms[1:T - 1, d - 1] * (-1 * X[1:T - 1, d - 2] * dt)) + \
                        (2 * terms[1:T - 1, 0] * (-1 + dt)) + \
                        (2 * terms[1:T - 1, 1] * (-1) * (X[1:T - 1, 2] - X[1:T - 1, d - 1]) * dt) + \
                        (2 * terms[1:T - 1, 2] * X[1:T - 1, 1] * dt) + (2 * terms[0:T - 2, 0])

    dervs[0, 0] = (2 * terms[0, d - 1] * (-1 * X[0, d - 2] * dt)) + \
                  (2 * terms[0, 0] * (-1 + dt)) + \
                  (2 * terms[0, 1] * (-1) * (X[0, 2] - X[0, d - 1]) * dt) + \
                  (2 * terms[0, 2] * X[0, 1] * dt)

    dervs[T - 1, 0] = (2 * terms[T - 2, 0])

    # dimension 1

    dervs[1:T - 1, 1] = (2 * terms[1:T - 1, 0] * (-1 * X[1:T - 1, d - 1] * dt)) + \
                        (2 * terms[1:T - 1, 1] * (-1 + dt)) + \
                        (2 * terms[1:T - 1, 2] * (-1) * (X[1:T - 1, 3] - X[1:T - 1, 0]) * dt) + \
                        (2 * terms[1:T - 1, 3] * X[1:T - 1, 2] * dt) + (2 * terms[0:T - 2, 1])

    dervs[0, 1] = (2 * terms[0, 0] * (-1 * X[0, d - 1] * dt)) + \
                  (2 * terms[0, 1] * (-1 + dt)) + \
                  (2 * terms[0, 2] * (-1) * (X[0, 3] - X[0, 0]) * dt) + \
                  (2 * terms[0, 3] * X[0, 2] * dt)

    dervs[T - 1, 1] = (2 * terms[T - 2, 1])

    # dimension d-2

    dervs[1:T - 1, d - 2] = (2 * terms[1:T - 1, d - 3] * (-1 * X[1:T - 1, d - 4] * dt)) + \
                            (2 * terms[1:T - 1, d - 2] * (-1 + dt)) + \
                            (2 * terms[1:T - 1, d - 1] * (-1) * (X[1:T - 1, 0] - X[1:T - 1, d - 3]) * dt) + \
                            (2 * terms[1:T - 1, 0] * X[1:T - 1, d - 1] * dt) + (2 * terms[0:T - 2, d - 2])

    dervs[0, d - 2] = (2 * terms[0, d - 3] * (-1 * X[0, d - 4] * dt)) + \
                      (2 * terms[0, d - 2] * (-1 + dt)) + \
                      (2 * terms[0, d - 1] * (-1) * (X[0, 0] - X[0, d - 3]) * dt) + \
                      (2 * terms[0, 0] * X[0, d - 1] * dt)

    dervs[T - 1, d - 2] = (2 * terms[T - 2, d - 2])

    # dimension d-1

    dervs[1:T - 1, d - 1] = (2 * terms[1:T - 1, d - 2] * (-1 * X[1:T - 1, d - 3] * dt)) + \
                            (2 * terms[1:T - 1, d - 1] * (-1 + dt)) + \
                            (2 * terms[1:T - 1, 0] * (-1) * (X[1:T - 1, 1] - X[1:T - 1, d - 2]) * dt) + \
                            (2 * terms[1:T - 1, 1] * X[1:T - 1, 0] * dt) + (2 * terms[0:T - 2, d - 1])

    dervs[0, d - 1] = (2 * terms[0, d - 2] * (-1 * X[0, d - 3] * dt)) + \
                      (2 * terms[0, d - 1] * (-1 + dt)) + \
                      (2 * terms[0, 0] * (-1) * (X[0, 1] - X[0, d - 2]) * dt) + \
                      (2 * terms[0, 1] * X[0, 0] * dt)

    dervs[T - 1, d - 1] = (2 * terms[T - 2, d - 1])

    g = dervs + 2 * lam * (X - X0)
    g = g.flatten('F')
    return g



def lorenz96_param_obj(params, X, dt):
    '''
    This function returns the objective value in eq. (8) of the paper. The function will be passed to the
    scipy.optimize.minimize() function as a parameter.

    Input:
        params: 1-dimensional parameter.
        X: T*d numpy vector, which contains the states.
        dt: Time interval between the state observations.
    Output:
        objval: The objective value
    '''

    T = X.shape[0]
    d = X.shape[1]
    F= params[0]
    terms = np.zeros((T-1,d))

    temp1 = ((X[0:T-1,3:d]-X[0:T-1,0:d-3])*X[0:T-1,1:d-2] - X[0:T-1,2:d-1] + F)*dt
    terms[:,2:d-1] = X[1:T,2:d-1] - X[0:T-1,2:d-1] - temp1

    temp1 = ((X[0:T-1,1]-X[0:T-1,d-2])*X[0:T-1,d-1] - X[0:T-1,0] + F)*dt
    terms[:,0] = X[1:T,0] - X[0:T-1,0] - temp1

    temp1 = ((X[0:T-1,2]-X[0:T-1,d-1])*X[0:T-1,0] - X[0:T-1,1] + F)*dt
    terms[:,1] = X[1:T,1] - X[0:T-1,1] - temp1

    temp1 = ((X[0:T-1,0]-X[0:T-1,d-3])*X[0:T-1,d-2] - X[0:T-1,d-1] + F)*dt
    terms[:,d-1] = X[1:T,d-1] - X[0:T-1,d-1] - temp1

    objval = np.sum((terms)**2)
    return objval

def lorenz96_param_grad(params, X, dt):
    '''
    This function returns the gradient with respect to the parameters in the eq. (8) of the paper.
    The function will be passed to the scipy.optimize.minimize() function as a parameter.

    Input:
        please look at the previous function 'lorenz_param_obj()' for the input variables.

    Output:
       g: It has the same size as the params and contains the gradients.
    '''
    
    T = X.shape[0]
    d = X.shape[1]
    F= params[0]
    terms = np.zeros((T-1,d))
    temp1 = ((X[0:T-1,3:d]-X[0:T-1,0:d-3])*X[0:T-1,1:d-2] - X[0:T-1,2:d-1] + F)*dt
    terms[:,2:d-1] = X[1:T,2:d-1] - X[0:T-1,2:d-1] - temp1

    temp1 = ((X[0:T-1,1]-X[0:T-1,d-2])*X[0:T-1,d-1] - X[0:T-1,0] + F)*dt
    terms[:,0] = X[1:T,0] - X[0:T-1,0] - temp1

    temp1 = ((X[0:T-1,2]-X[0:T-1,d-1])*X[0:T-1,0] - X[0:T-1,1] + F)*dt
    terms[:,1] = X[1:T,1] - X[0:T-1,1] - temp1

    temp1 = ((X[0:T-1,0]-X[0:T-1,d-3])*X[0:T-1,d-2] - X[0:T-1,d-1] + F)*dt
    terms[:,d-1] = X[1:T,d-1] - X[0:T-1,d-1] - temp1

    ga = np.sum(terms) * (-2*dt)
    res = np.array([ga])
    return res