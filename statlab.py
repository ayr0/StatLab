import scipy.stats as st
import scipy as sp
import scipy.linalg as la

class StatLab(object):
    def __init__(self, response, ivars):
        self.response = Y
        self.ivars = X
        

    def MLE(self):
        r"""
        Calculate the Maximum Likelyhood Estimation
        """
        pass

    def MSE(self):
        """Calculate the MSE"""

        k = min(self.X.shape)-1
        n = max(self.Y.shape)

        


        H = X.dot(la.inv(X.T.dot(X)).dot(X.T))
        yhy=Y.T.dot((sp.eye(H.shape[0])-H).dot(Y))
        
        if yhy.shape == (1,n):
            return (1.0/(n-k-1))*sp.dot(yhy, yhy.T)
        else:
            return (1.0/(n-k-1))*sp.dot(yhy.T, yhy)

    def _SSE(self):
        fy = beta_hat().dot(self.X)
        return (self.Y-fy).T.dot(self.Y-fy)
        
    def beta_hat(self):
        return self._beta_hat(self.ivars, self.response)
        
    def _test_statistics(self, beta_hat, beta):
        return B[bvar,0]/float(covar[bvar,bvar])
        