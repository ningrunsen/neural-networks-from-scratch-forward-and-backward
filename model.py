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

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
   
    
  

    w,b  = weight_init_fn(in_dim,out_dim)
    params = {
      'W':w,
      'b':b
    }
    def forward(x):
      w = params['W']
      b = params['b']
      y = x @ w + b

      return (y,x)
    
    def backward(dout,cache):

      dx = dout @ w.T
      dw = cache.T @ dout
      db = np.sum(dout,axis=0)
      grads={
        'W':dw,
        'b':db
      }
      return dx,grads
    return {
      'params':params,
      'forward':forward,
      'backward':backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    

    def forward(x):
      return (np.where(x>0,x,0),x)

    def backward(dout, cache):
        dx = dout *(cache>0)

        return dx,{}


    return {
      'forward':forward,
      'backward':backward,
      'params':{}

    }

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

