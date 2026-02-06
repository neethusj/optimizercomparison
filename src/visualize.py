import matplotlib.pyplot as plt
import numpy as np

#Plot loss Vs Iterations
def plot_loss_iterations(result_dict,title="Loss Vs Iterations"):
    fname=result_dict["Function"]
    # plt.figure()
    for opt_name,data in result_dict.items():
        if opt_name=='Function': continue
        loss=data["loss"]
        plt.plot(loss,label=opt_name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title+" :"+fname.__name__)
    plt.grid()
    plt.legend()
    #plt.savefig("results/plots/"+fname.__name__+"_loss.png")
    #plt.show()

# Trajectory Map for 1D functions
def plot_trajectory_1d(result_dict,title="Trajectory 1d"):
    func=result_dict["Function"]
    x_vals=np.linspace(-6,6,400)
    y_vals=[func(x) for x in x_vals]
    # plt.figure()
    s=100
    plt.plot(x_vals,y_vals,'k-',label="Loss Curve")
    for opt_name,data in result_dict.items():
        if opt_name=="Function":continue
        traj=data["traj"]
        loss=data["loss"]
        plt.scatter(traj,loss,label=opt_name,s=s)
        s=s-20
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title+" :"+func.__name__)
    plt.legend()
    #plt.savefig("results/plots/"+func.__name__+"_trajectory.png")
    #plt.show()

def plot_trajectory_2d(result_dict,title="Trajectory 2D"):
    func = result_dict["Function"]
    # Collect all trajectory points to set plotting range dynamically
    all_points = []
    for opt_name, data in result_dict.items():
        if opt_name == "Function": continue
        all_points.extend(data["traj"])
    all_points = np.array(all_points)
    min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
    min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
    # Expand range slightly for better visualization
    x = np.linspace(min_x - 1, max_x + 1, 400)
    y = np.linspace(min_y - 1, max_y + 1, 400)
    X, Y = np.meshgrid(x, y)
    # Evaluate function on grid
    Z = func(X, Y)
    # Plot contour map
    #plt.figure(figsize=(7,6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    # Overlay optimizer trajectories 
    for opt_name, data in result_dict.items():
         if opt_name == "Function": continue 
         traj = np.array(data["traj"]) 
         plt.plot(traj[:,0], traj[:,1], 'o-', label=opt_name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title+" :"+func.__name__)
    plt.legend()
    #plt.show()

def plot_dual(result_dict,plot_type='1d',title="Dual Plot"):
    func = result_dict["Function"]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    #Left:Loss vs Iterations
    #Redirect plotting to ax1
    plt.sca(axes[0])
    plot_loss_iterations(result_dict, title="Loss vs Iterations")
    # Right subplot: Trajectory 
    plt.sca(axes[1])
    if plot_type == '1d': 
        plot_trajectory_1d(result_dict, title="Trajectory 1D") 
    elif plot_type == '2d': 
        plot_trajectory_2d(result_dict, title="Trajectory 2D")
    else: 
        raise ValueError("plot_type must be '1d' or '2d'") 
    plt.tight_layout()
    plt.savefig("results/plots/"+func.__name__+"_dualplot.png")
    plt.show()

    


