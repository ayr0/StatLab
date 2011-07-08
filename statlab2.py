import scipy.stats as st
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

class StatLab(object):

    def __init__(self, X, Y):
        self.Y = Y
        self.X = X
        self.alpha = 0.05
        
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

        self.hypothesis = sp.zeros(self.Xdim)
        
        self.tstats = sp.zeros(self.Xdim)
        for x in range(self.Xdim):
            self.tstats[x] = (self.beta_hat[x]-self.hypothesis[x])/(self.MSE*sp.math.sqrt(self.variance[x,x]))
        
        
        self.drule=st.t.ppf(1-self.alpha/2.0, self.ntrials-self.Xdim)
        self.pvals = 2.0*(1-st.t.cdf(abs(self.tstats), self.ntrials-self.Xdim))
    
    def analyze(self):
        print "    Alpha: %.3f" % self.alpha
        print "    Decision rule: %.5f" %self.drule
        print "    beta             dev              tstats           pvals"
        print sp.vstack((self.beta_hat.T,sp.sqrt(sp.diag(self.variance)), self.tstats, self.pvals)).T
        return 
        
    def plotX(self, n):
        plt.plot(self.X[:,n],self.Y,'o')
        plt.show()

    def confInt(self, n):
        x = sp.math.sqrt(self.variance[n,n])*self.drule
        lowBound = self.beta_hat[n] - x
        upBound = self.beta_hat[n] + x
        print "The %d%% confidence interval for X%d is from %f.5 to %f.5" %((1-self.alpha)*100, n, lowBound, upBound)
        
    def fitVsRes(self):
        fitted = self.X.dot(self.beta_hat)
        plt.plot(fitted,self._residuals,'o')
        plt.show()
        
    def histRes(self):
        plt.hist(self._residuals)
        plt.show()
    
    def QQ(self):
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
            return the_vars, rm_vars
        elif dir == 'f':
            #Perform forward stepwise regression
            raise NotImplementedError("This ain't implemented yet!")
