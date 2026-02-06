from optimizers import (gradient_descent_optimizer,momentum_optimizer,
                           rmsprop_optimizer,adam_optimizer)
import numpy as np

def run_experiment(func,grad,init):

    #compute trajectories produced by the optimizers
    traj_gd=gradient_descent_optimizer(grad,init)
    traj_mom=momentum_optimizer(grad,init)
    traj_rms=rmsprop_optimizer(grad,init)
    traj_adam=adam_optimizer(grad,init)

    # Compute Loss for the values in the trajectory
    loss_gd   = [func(*x) if np.ndim(x) > 0 else func(x) for x in traj_gd]
    loss_mom  = [func(*x) if np.ndim(x) > 0 else func(x) for x in traj_mom]
    loss_rms  = [func(*x) if np.ndim(x) > 0 else func(x) for x in traj_rms]
    loss_adam = [func(*x) if np.ndim(x) > 0 else func(x) for x in traj_adam]


    #create  a dictionary with optimizer name and loss function values

    result_dict={
        "Gradient Descent":{"traj":traj_gd,"loss":loss_gd},
        "Momentum":{"traj":traj_mom,"loss":loss_mom},
        "Rms Prop":{"traj":traj_rms,"loss":loss_rms},
        "Adam":{"traj":traj_adam,"loss":loss_adam},
        "Function":func
    }

    return result_dict