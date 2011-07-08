import scipy.stats as st
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

class StatLab(object):

    def __init__(self, X, Y, alpha=0.05):
        self.Y = Y
        self.X = X
        self.alpha = alpha

        #first column is always intercept
        self.vtags = {0:'intercept'}
        
        self.calc()
        
    def calc(self):     
        self.Xdim = self.X.shape[1] #Number of progressors plus one
        self.ntrials = self.X.shape[0] #Number of trials
        self.Ymean = self.Y.sum()/float(self.ntrials)
        
        self.beta_hat = la.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.Y))
        self._projection = self.X.dot(self.beta_hat)
        self._residuals = self.Y - self._projection

        self.SSE = self._residuals.T.dot(self._residuals)
        self.MSE = self.SSE/(self.ntrials - self.Xdim)

        self.variance = la.inv(self.MSE*self.X.T.dot(self.X))
        self.deviation = sp.sqrt(sp.diag(self.variance))

        self.hypothesis = sp.zeros(self.Xdim)
		self.tstats = sp.diag((self.beta_hat-self.hypothesis)/(self.MSE*self.deviation))
        
        
        self.drule=st.t.ppf(1-self.alpha/2.0, self.ntrials-self.Xdim)
        self.pvals = 2.0*(1-st.t.cdf(abs(self.tstats), self.ntrials-self.Xdim))

    def tag(self, label, index=None):
        """Used to attach a tag to a random variable in X
        label is expected to be a list of strings,
        but if index is specified, it only has to be string"""

        if isinstance(label, (tuple, list)):
            #we have  a list
            for ind in xrange(len(label)):
                self.vtags[ind+1]=label[ind]

        elif isinstance(label, str):
            if index is None:
                raise IndexError("a variable index must be specified")
            else:
                #we have a string
                self.vtags[index]=label
                self.vtags
            
    def pickle_dump(self, filename):
        """Dump everything to a file"""
        from pickle import dump
        with open(filename, "w") as f:
            dump(self, f)
            
    def pickle_load(self, filename):
        """Read a pickled StatLab object"""
        from pickle import load
        with open(filename, 'r') as f:
            a = load(f)
        return f
        
    def analyze(self, sw=100):
        #Recalculate everything
        self.calc()

        vals = sp.vstack((self.beta_hat.T,self.deviation, self.tstats, self.pvals)).T
        
        
        print "Analyze".center(sw, "=")
        print ("\tAlpha = %.3f" % self.alpha)
        print ("\tDecision rule = %.5f" % self.drule)
        print ("\tR**2 adjusted = %.5f" % self.r2a())

		print "\n\tBeta\tDev\tT-stats\tP-vals"
        for irow in xrange(self.Xdim):
            if self.vtags.has_key(irow):
                label = self.vtags[irow][:7]
            else:
                label = str(irow)
            print "%s\t" % label,"\t".join([("%.5f"%v)[:7] for v in vals[irow,:]])
            
    def plotX(self, n):
        plt.plot(self.X[:,n],self.Y,'o')
        plt.show()

	def confIntParam(self, n):
		"""Calculate the confidence interval for the nth parameter"""
        x = sp.math.sqrt(self.variance[n,n])*self.drule
        lowBound = self.beta_hat[n] - x
        upBound = self.beta_hat[n] + x
        print "The %d%% confidence interval for X%d is from %f.5 to %f.5" %((1-self.alpha)*100, n, lowBound, upBound)
        
	def confIntObs(self, x):
		"""Prints the prediction, confidence interval and prediction interval
		for a given observation."""
		self.calc()
		x = list(x)
		if len(x) != self.X[0,1:].size:
			raise ValueError("Incorrect observation size")
		x.insert(0,1)
		x = sp.asarray(x)
		pred = x.dot(self.beta_hat)
		thing = self.MSE*sp.math.sqrt(x.dot(self.variance.dot(x.T)))
		thing1 = self.MSE*sp.math.sqrt(x.dot(self.variance.dot(x.T))+1)
		tdist = st.t.ppf(1-self.alpha, self.ntrials-self.Xdim)
		print "The prediction is %.5f" %pred
		print "[Confidence] The %d%% confidence interval is from %.5f to %.5f" \
			%(100-self.alpha*100,pred-thing*tdist,pred+thing*tdist)
		print "[Prediction] The %d%% prediction interval is from %.5f to %.5f" \
			%(100-self.alpha*100,pred-thing1*tdist,pred+thing1*tdist)
	
    def fitVsRes(self):
        """Plot fitted values vs. residuals"""
        fitted = self.X.dot(self.beta_hat)
        plt.plot(fitted,self._residuals,'o')
        plt.show()
        
    def histRes(self):
        """Plot histogram of residuals"""
        plt.hist(self._residuals)
        plt.show()
    
    def QQ(self):
        raise NotImplementedError("This ain't implemented yet!")
    
    def IWLS(self, niter=50):
        """Perform Iterative Weighted Least Squares"""
        
        def wbeta_hat(W):
            return la.inv(self.X.T.dot(W.dot(self.X))).dot(self.X.T.dot(W.dot(self.Y)))

        vi0 = 1.0/self._residuals**2
        W0 = sp.diag(vi)
        for i in xrange(50):
            pass
            
        
    
    def r2a(self):
        sum1 = sp.sum((self.Y - self._projection)**2)
        sum2 = float(sp.sum((self.Y - self.Ymean)**2))
        return (1 - (sum1*(self.ntrials-1))/(sum2*(self.ntrials - self.Xdim)))
        
    def _backStepWise(self):
        
        r0 = self.r2a()
        x = sp.array([True]*self.Xdim)
        r = [0]
        ichanges = []
        for i in xrange(1,self.Xdim):
            xc = x.copy()
            xc[i] = False
            Xtest = self.X[:,xc]
            test = StatLab(Xtest.copy(),self.Y.copy())
            r.append(test.r2a())
        imaxr = r.index(max(r))
        if r0 < max(r):
            x[imaxr] = False
            self.X = self.X[:,x]
            self.calc()
            ichanges.append(imaxr)
            ichanges.extend(self._backStepWise())
            return ichanges
        else:
            return [None]

    def stepwise_regr(self, dir='b'):
        """Perform stepwise regression on the data set"""

        if dir == 'b':
            #Perform backward stepwise regression
            the_vars = range(self.Xdim)
            changes = self._backStepWise()
        
            rm_vars = [the_vars.pop(x) for x in changes if x is not None]
            self.the_vars = the_vars
            self.rm_vars = rm_vars
            return the_vars, rm_vars
        elif dir == 'f':
            #Perform forward stepwise regression
            raise NotImplementedError("This ain't implemented yet!")
