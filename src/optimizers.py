import numpy as np

#--------------------------------------------------------------------
# Gradient Descent
#--------------------------------------------------------------------

def gradient_descent_optimizer(grad,init,alpha=0.01,iterations=1000,clip_value=10):
    """
    grad:gradient function
    init:initial point
    alpha:learning rate
    iterations:Number of steps the optimizer takes
    Returns:
    trajectory:Array of points visited during optimization
    """
    x=np.array(init,dtype=float)
    trajectory=[x.copy()]
    for i in range(iterations):
        gradient = grad(*x) if np.ndim(x) > 0 else grad(x)
        #gradient clipping to prevent exploding gradient
        gradient = np.clip(gradient, -clip_value, clip_value)
        #position update
        x=x-(alpha*gradient)
        trajectory.append(x.copy())
    return np.array(trajectory)

#-----------------------------------------------------------------------------------
#  Momentum Gradient Descent
#----------------------------------------------------------------------------------

def momentum_optimizer(grad,init,alpha=0.01,beta=0.9,iterations=1000,clip_value=10):
    """
    grad:gradient function
    init:initial point
    alpha:learning rate
    beta:Momentum coefficient
    iterations:Number of steps the optimizer takes

    Returns:
    trajectory:Array of points visited during optimization
    """
    x=np.array(init,dtype=float)
    v=np.zeros_like(x)
    trajectory=[x.copy()]
    
    for i in range(iterations):
        gradient = grad(*x) if np.ndim(x) > 0 else grad(x)
        #gradient clipping to prevent exploding gradient
        gradient = np.clip(gradient, -clip_value, clip_value)
        #velocity update
        v=beta*v+alpha*gradient
        #position update
        x=x-v
        trajectory.append(x.copy())
    return np.array(trajectory)

#----------------------------------------------------------------------
# RMSProp Optimizer(Root Mean Square Propagation)-Adaptive Learning rate
#---------------------------------------------------------------------
def rmsprop_optimizer(grad,init,alpha=0.01,beta=0.9,eps=1e-8,iterations=1000,clip_value=10):
    """
    grad:gradient function
    init:initial point
    alpha:learning rate
    beta:decay rate for moving average of squared gradients
    eps:small constant to avoid division by 0
    iterations:Number of steps the optimizer takes

    Returns:
    trajectory:Array of points visited during optimization
    """
    
    x=np.array(init,dtype=float)
    s=np.zeros_like(x) # Initialize running average of squared gradients
    trajectory=[x.copy()]

    for i in range(iterations):
        gradient = grad(*x) if np.ndim(x) > 0 else grad(x)
        #gradient clipping to prevent exploding gradient
        gradient = np.clip(gradient, -clip_value, clip_value)
        s=beta*s+(1-beta)*(gradient**2)        # Update running average of squared gradients
        x=x-alpha*gradient/(np.sqrt(s)+eps)    # parameter update
        trajectory.append(x.copy())

    return np.array(trajectory)



#-------------------------------------------------------------
#  Adam optimizer
#  Combines momentum (first moment) and adaptive learning rates (second moment).
#--------------------------------------------------------------

def adam_optimizer(grad,init,alpha=0.01,iterations=1000,beta1=0.9,beta2=0.999,eps=1e-8,clip_value=10):
    """
    grad:gradient function
    init:initial point
    alpha:learning rate
    beta1:decay rate for first moment
    beta2:decay rate for second moment
    eps:small constant to avoid division by 0
    iterations:Number of steps the optimizer takes

    Returns:
    trajectory:Array of points visited during optimization
    """
    
    x=np.array(init,dtype=float)
    m=np.zeros_like(x)
    v=np.zeros_like(x)
    trajectory=[x.copy()]
    for t in range(1,iterations+1):
        gradient = grad(*x) if np.ndim(x) > 0 else grad(x)
        #gradient clipping to prevent exploding gradient
        gradient = np.clip(gradient, -clip_value, clip_value)
        m=beta1*m+(1-beta1)*gradient           #first moment
        v=beta2*v+(1-beta2)*(gradient**2)      #second moment
        m_hat=m/(1-beta1**t)     #bias correction for first moment
        v_hat=v/(1-beta2**t)     #bias correction for second moment
        x=x-alpha*m_hat/(np.sqrt(v_hat)+eps)    #parameter update
        trajectory.append(x.copy())
    return np.array(trajectory)
