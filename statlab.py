import scipy.stats as st
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

class StatLab(object):

	def __init__(self, X, Y):
		self.response = Y
		self.ivars = X
		self.alpha = 0.05
		self.calc()
			
	def calc(self):
		self.Xdim = self.ivars.shape[1] #Number of progressors plus one
		self.ndim = self.ivars.shape[0] #Number of trials
		
		self.beta_hat = la.inv(self.ivars.T.dot(self.ivars)).dot(self.ivars.T.dot(self.response))
		self._projection = self.ivars.dot(self.beta_hat)
		self._residuals = self.response - self._projection

		self.SSE = self._residuals.T.dot(self._residuals)
		self.MSE = self.SSE/(self.ndim - self.Xdim)

		self.variance = la.inv(self.MSE*self.ivars.T.dot(self.ivars))

		self.hypothesis = sp.zeros(self.Xdim)
		
		self.tstats = sp.zeros(self.Xdim)
		for x in range(self.Xdim):
			self.tstats[x] = (self.beta_hat[x]-self.hypothesis[x])/(self.MSE*sp.math.sqrt(self.variance[x,x]))
		
		
		self.drule=st.t.ppf(1-(self.alpha/2.0), self.ndim-self.Xdim)
		self.pvals = 2.0*(1-st.t.cdf(abs(self.tstats), self.ndim-self.Xdim))
	
	def analyze(self):
		print "    Alpha: %.3f" % self.alpha
		print "    Decision rule: %.5f" %self.drule
		print "    Beta             Dev              T-stats           P-vals"
		print sp.vstack((self.beta_hat.T,sp.sqrt(sp.diag(self.variance)), self.tstats, self.pvals)).T
		return 
		
	def plotX(self, n):
		plt.plot(self.ivars[:,n],self.response,'o')
		plt.show()

	def confInt(self, n, confidence=None):
		if confidence is not None:
			print "For %d%% confidence of X%d, alpha needs to be %.3f" % (confidence*100.0, n, 1.0-confidence)
			return 1.0-confidence
		else:	 
			x = sp.math.sqrt(self.variance[n,n])*self.drule
			lowBound = self.beta_hat[n] - x
			upBound = self.beta_hat[n] + x
			print "The %d%% confidence interval for X%d is from %f.5 to %f.5" %((1-self.alpha)*100, n, lowBound, upBound)