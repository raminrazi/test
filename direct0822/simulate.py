import numpy as np
from scipy.integrate import ode
import sys
sys.path.append('./ODEs/')
from fitzhugh import fitzhugh_ode
from rossler import rossler_ode
from lotka_volterra import  lotka_volterra_ode
from lorenz96 import lorenz96_ode

def simulate(ODE_str, x0, true_param, end_t, dt, noise_var):
    '''
    This function creates clean states and noisy observations for the ODEs.

    Input:
      ode_str: Name of the model as a string. Set ODE_str to 'fitzhugh' or 'lotka_volterra' or 'rossler' or 'lorenz96'
      x0: A d-dimensional list that contains initial state at time 0.
      true_param: A p-dimensional list that contains the true parameters of ODE.
      end_t: The final time of the simulation. The start time is 0.
      dt: The time interval between samples.
      noise_var: The variance of the Gaussian noise. The noise will be used in creating noisy observations.

    Output:
      X: T*d numpy array that contains the clean states.
      Y: T*d numpy array that contains the noisy observations.
      dt: Time interval between samples.
    '''

    # change the string into a function.
    ode_fun = eval(ODE_str+'_ode')

    d = len(x0)
    t0 = 0
    r = ode(ode_fun).set_integrator('dopri5').set_f_params(true_param)
    r.set_initial_value(x0, t0)
    T = int((end_t/dt) + 1)
    X = np.empty((T,d))
    X[0,:] = x0
    idx=1
    while idx < T:
        r.integrate(r.t+dt)
        X[idx, :] = r.y
        idx = idx + 1

    Y = X + (np.random.normal(0,noise_var,(T,d)))

    return X,Y,dt



