from scipy import exp, log, isscalar, logical_and
from scipy.special import gammaln

def gammapdf(X, shape, scale):
    return exp((shape-1.0)*log(X) - (X/(1.0*scale)) + gammaln(shape) + (shape * log(scale)))

def betapdf(X,a,b):
    #operate only on x in (0,1)
    if isscalar(X):
        if X<=0 or X>=1:
            raise ValueError("X must be in the interval (0,1)")
        else:
            x=X
    else:
        goodx = logical_and(X>0, X<1)
        x = X[goodx].copy()

    loga = (a-1.0)*log(x)
    logb = (b-1.0)*log(1.0-x)

    return exp(gammaln(a+b)-gammaln(a)-gammaln(b)+loga+logb)
