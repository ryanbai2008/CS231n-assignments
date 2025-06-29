from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
      scores = np.matmul(X[i], W)
      instability = np.repeat(np.max(scores), num_class)
      scores -= instability
      scores = np.power(math.e, scores)
      sum = np.sum(scores)
      scores /= sum
      loss += -(math.log(scores[y[i]]))
      for j in range(num_class):
          if j == y[i]:
            dW[:,j] += X[i] * (scores[y[i]] - 1)
          else:
             dW[:, j] += X[i] * (scores[j])
    
    loss = (loss / num_train) + reg * np.sum(np.square(W))
    dW = (dW / num_train) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    
    #scores
    scores = np.matmul(X, W)
    maxScores = np.reshape(np.repeat(np.max(scores, axis = 1), scores.shape[1]), (scores.shape[0], scores.shape[1]))
    scores = scores - maxScores
    scores = np.exp(scores)
    rowSum = np.reshape(np.repeat(np.sum(scores, axis = 1), scores.shape[1]), (scores.shape[0], scores.shape[1]))
    scores = scores / rowSum

    #loss
    loss += -(np.sum(np.log(scores[range(num_train), y])) / num_train) + reg * np.sum(np.square(W))

    #gradient
    # correctClass = np.zeros((num_train, num_class))
    # correctClass[range(num_train), y] = scores[range(num_train), y]
    scores[range(num_train), y] -= 1
    dW = (np.matmul(X.T, scores) / num_train) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
