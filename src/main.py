from experiments import run_experiment
from visualize import plot_loss_iterations,plot_trajectory_1d,plot_trajectory_2d,plot_dual

from functions import (quadratic,grad_quadratic,non_convex,gradient_nonconvex,
                       rosenbrock,gradient_rosenbrock,rastrigin,
                       gradient_rastrigin,himmelblau,gradient_himmelblau)
import numpy as np


result_dict=run_experiment(quadratic,grad_quadratic,5.0)
# plot_loss_iterations(result_dict)
# plot_trajectory_1d(result_dict,title="Quadratic trajectories")
plot_dual(result_dict, plot_type='1d', title="Quadratic")


result_dict=run_experiment(non_convex,gradient_nonconvex,5.0)
# plot_loss_iterations(result_dict)
# plot_trajectory_1d(result_dict,title="Non convex trajectories")
plot_dual(result_dict, plot_type='1d', title="Non Convex")

result_dict=run_experiment(rosenbrock,gradient_rosenbrock,np.array([1.2, 1.2])  )
# plot_loss_iterations(result_dict)
# plot_trajectory_2d(result_dict, title="Rosenbrock Trajectories")
plot_dual(result_dict, plot_type='2d', title="Rosenbrock")

result_dict=run_experiment(rastrigin,gradient_rastrigin,np.array([1.0, 1.0]))
# plot_loss_iterations(result_dict)

# result_dict=run_experiment(rastrigin,gradient_rastrigin,np.array([0.5, 0.5]))
# plot_loss_iterations(result_dict)
# plot_trajectory_2d(result_dict, title="Rastrigin Trajectories")
plot_dual(result_dict, plot_type='2d', title="Rastrigin")

result_dict=run_experiment(himmelblau,gradient_himmelblau,np.array([3.0, 2.0]))
# plot_loss_iterations(result_dict)
# plot_trajectory_2d(result_dict, title="Himmelblau Trajectories")
plot_dual(result_dict, plot_type='2d', title="Himmelblau")