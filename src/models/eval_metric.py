import numpy as np

def mcrmse(ytrue, ypred):
    s = 0
    
    for j in range(len(ytrue)):
        diff = (ytrue[j] - ypred[j]) ** 2
        diff_sum = np.sum(diff)
        s = s + np.sqrt((1/len(ytrue[j])) * diff_sum)
    
    return (1/len(ytrue)) * s