from builtins import range
from pathlib import WindowsPath
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # print(num_classes)
    # print(num_train)
    for i in range(num_train):
        # Forward pass
        scores = X[i].dot(W)
        # Shift scores to prevent numerical instability
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # Compute loss
        correct_logprob = -np.log(probs[y[i]])
        loss += correct_logprob

        # Compute gradient
        dscores = probs.copy()
        dscores[y[i]] -= 1
        dW += np.outer(X[i], dscores)

    # Average loss and gradient
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # W=sW.reshape(1, -1)
    # X=X.T.reshape(1,-1)
    
    # print(X.shape)
    # print(W.shape)
    scores = np.dot(X, W)
    
    # Subtract the maximum score for numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # Compute softmax scores
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute the loss for each example
    correct_logprobs = -np.log(probs[range(num_train), y])
    data_loss = np.sum(correct_logprobs) / num_train
    
    # Add regularization to the loss
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # Compute gradients
    dscores = probs.copy()
    dscores[range(num_train), y] -= 1
    dscores /= num_train

    dW = np.dot(X.T, dscores)
    dW += 0.5*reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
