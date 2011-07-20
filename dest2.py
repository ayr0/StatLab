from scipy import size,linspace,mean,median
from matplotlib.pyplot import plot,show
from scipy.stats.kde import gaussian_kde

def quickBeta(x,bx,y,by,num=10000):
	from scipy.stats import beta
	return dest(beta.rvs(x[0]+bx[0],x[1]+bx[1],size=num)-beta.rvs(y[0]+by[0],y[1]+by[1],size=num))

class dest(object):
	"""Density estimator"""

	def __init__(self, rvs, inc=1000):
		self.rvs = rvs.copy()
		self.rvs.sort()
		self.size = size(self.rvs)
		self.kde = gaussian_kde(self.rvs)
		self.lowb = self.rvs[0] - (self.ppf(.1)-self.ppf(0.))
		self.upb = self.rvs[self.size-1] + (self.ppf(.1)-self.ppf(0.))
		self.inc = inc
		self.x = linspace(self.lowb,self.upb,self.inc)
	
	def pdf(self,x):
		if size(x)==1:
			return self.kde(x)[0]
		else:
			return self.kde(x)
		
	def ppf(self,x):
		if x<0 or x>1:
			raise ValueError("Not between 0 and 1")
		return self.rvs[int(x*(self.size-1))]
		
	def cdf(self,x):
		return size(self.rvs[self.rvs<x])/float(self.size)
		
	def graph(self):
		plot(self.x,self.kde(self.x))
		show()
	
	def postInt(self,x):
		x = (1-x)/2.
		return self.ppf(x),self.ppf(1-x)
	
	def mean(self):
		return mean(self.rvs)
		
	def median(self):
		return median(self.rvs)
		
	def mode(self):
		return self.x[self.kde(self.x).argmax()]
		
	