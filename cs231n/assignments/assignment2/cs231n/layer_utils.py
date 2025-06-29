from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def fc_net_forward(X, w, b, gamma, beta, bn_param, dropout_param, last=False):
    if last:
        out_fc, cache_fc = affine_forward(X, w, b)
        cache = {"fc" : cache_fc, "relu" : None, "bn" : None}
        return out_fc, cache
    else:
        if bn_param is not None:
            out_fc, cache_fc = affine_forward(X, w, b)
            out_relu, cache_relu = relu_forward(out_fc)
            out_bn, cache_bn = batchnorm_forward(out_relu, gamma, beta, bn_param)
            out_dropout, cache_dropout = dropout_forward(out_bn)
            cache = {"fc" : cache_fc, "relu" : cache_relu, "bn" : cache_bn, "dropout" : cache_dropout}
            return out_dropout, cache
        elif bn_param is None and dropout_param != None:
            out_fc, cache_fc = affine_forward(X, w, b)
            out_relu, cache_relu = relu_forward(out_fc)
            out_dropout, cache_dropout = dropout_forward(out_relu, dropout_param)
            cache = {"fc" : cache_fc, "relu" : cache_relu, "bn" : None, "dropout" : cache_dropout}
            return out_dropout, cache
        elif bn_param is None:
            out_fc, cache_fc = affine_forward(X, w, b)
            out_relu, cache_relu = relu_forward(out_fc)
            cache = {"fc" : cache_fc, "relu" : cache_relu, "bn" : None}
            return out_relu, cache
        else: #LAYERNORM
            out_fc, cache_fc = affine_forward(X, w, b)
            out_relu, cache_relu = relu_forward(out_fc)
            out_ln, cache_ln = layernorm_forward(out_relu, gamma, beta, bn_param)
            cache = {"fc" : cache_fc, "relu" : cache_relu, "bn" : None, "ln" : cache_ln}
            return out_ln, cache

def fc_net_backward(dout, cache):
    if cache["bn"] is not None:
        dout, dgamma, dbeta = batchnorm_backward(dout, cache["bn"])
        dout = relu_backward(dout, cache["relu"])
        dout, dw, db = affine_backward(dout, cache["fc"])
        return dout, dw, db, dgamma, dbeta
    elif cache["bn"] is None and cache["relu"] is not None:
        dout = relu_backward(dout, cache["relu"])
        dout, dw, db = affine_backward(dout, cache["fc"])
        return dout, dw, db, None, None
    elif cache.get("ln") is not None: #LAYERNORM
        dout, dgamma, dbeta = layernorm_backward(dout, cache["ln"])
        dout = relu_backward(dout, cache["relu"])
        dout, dw, db = affine_backward(dout, cache["fc"])
        return dout, dw, db, dgamma, dbeta
    elif cache.get("dropout") is not None:
        dout = dropout_backward(dout, cache["dropout"])
        dout, dgamma, dbeta = batchnorm_backward(dout, cache["bn"])
        dout = relu_backward(dout, cache["relu"])
        dout, dw, db = affine_backward(dout, cache["fc"])
        return dout, dw, db, dgamma, dbeta
    else:
        dout, dw, db = affine_backward(dout, cache["fc"])
        return dout, dw, db, None, None

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
