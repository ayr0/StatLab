import scipy.stats as st
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

class StatLab(object):

	def __init__(self, X, Y):
		self.response = Y
		self.ivars = X
		self.Xdim = X.shape[1] #Number of progressors plus one
		self.ndim = X.shape[0] #Number of trials
		self.beta_hat = la.inv(X.T.dot(X)).dot(X.T.dot(Y))
		self.projection = X.dot(self.beta_hat)
		self.residuals = Y - self.projection
		self.SSE = self.residuals.T.dot(self.residuals)
		self.MSE = self.SSE/(self.ndim - self.Xdim)
		self.variance = la.inv(self.MSE*X.T.dot(X))
		self.hypothesis = sp.zeros(self.Xdim)
		self.alpha = 0.5
	
	def test_stats(self):
		tstats = sp.zeros(self.Xdim)
		for x in range(self.Xdim):
			tstats[x] = (self.beta_hat[x]-self.hypothesis[x])/(self.MSE*sp.sqrt(self.variance[x,x]))
		return tstats
		
	def dec_rule(self):
		return st.t.ppf(1-self.alpha/2.0, self.ndim-self.Xdim)
		
	def pvals(self):
		return 2.0*(1-st.t.cdf(abs(self.test_stats()), self.ndim-self.Xdim))
		
	def analyze(self):
		print "    Alpha: %.3f" % self.alpha
		print "    Decision rule: %.5f" %self.dec_rule()
		print "    beta             dev              tstats           pvals"
		print sp.vstack((self.beta_hat.T,sp.sqrt(sp.diag(self.variance)), self.test_stats(), self.pvals())).T
		return 

	def plotX(self, n):
		plt.plot(self.ivars[:,n],self.response,'o')
		plt.show()