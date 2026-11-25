"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    
    
    res = np.zeros_like(x)
    for pos in np.ndindex(x.shape):
        arr1 = x.copy()
        arr2 = x.copy()
        arr1[pos] +=eps
        arr2[pos]-=eps
        grad = (f(arr1)-f(arr2))/(2*eps)
        res[pos]=grad

    return res

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    
    error = np.abs(analytic_grad-numeric_grad)/(np.maximum(np.maximum(analytic_grad,numeric_grad),tol))

    return np.max(error)

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

