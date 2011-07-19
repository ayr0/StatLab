from scipy import size,linspace
from matplotlib.pyplot import plot,show
from scipy.stats.kde import gaussian_kde

class dest(object):
	"""Density estimator"""

	def __init__(self, rvs):
		self.rvs = rvs.copy()
		self.rvs.sort()
		self.size = size(self.rvs)
		self.kde = gaussian_kde(self.rvs)
	
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
		
	def graph(self,inc=1000):
		lowb = self.rvs[0] - (self.ppf(.1)-self.ppf(0.))
		upb = self.rvs[self.size-1] + (self.ppf(.1)-self.ppf(0.))
		x = linspace(lowb,upb,inc)
		plot(x,self.kde(x))
		show()
		
	