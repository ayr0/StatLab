import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def outlier():

    zero = sp.zeros((11,1))
    ones = sp.ones((11,1))

    Y1 = sp.vstack([sp.ones((44,1)), sp.ones((11,1)), sp.zeros((44,1))])
    Y2 = sp.vstack([sp.zeros((55,1)), sp.ones((44,1))])
    Y = sp.hstack((Y1,Y2))
    
    X1 = sp.random.multivariate_normal([-3,3], [[.1,.09],[.09,.1]], size=44)
    X2 = sp.random.multivariate_normal([3,0], [[1,0],[0,1]], size=11)
    X3 = sp.random.multivariate_normal([-4,4], [[.1,.09],[.09,.01]], size=44)

    X = sp.hstack((sp.ones((99,1)),sp.vstack((X1,X2,X3))))

    return Y, X

def stripe3():
    zero = sp.zeros((33,1))
    ones = sp.ones((33,1))

    Y1 = sp.vstack([ones, zero, zero])
    Y2 = sp.vstack([zero, ones, zero])
    Y3 = sp.vstack([zero, zero, ones])
    Y = sp.hstack((Y1, Y2, Y3))

    X1 = sp.random.multivariate_normal([-2,2], [[1,.8],[.8,1]], size=33)
    X2 = sp.random.multivariate_normal([2,-2], [[1,.8],[.8,1]], size=33)
    X3 = sp.random.multivariate_normal([0,0], [[1,.8],[.8,1]], size=33)
    X = sp.hstack((sp.vstack((ones,ones,ones)),sp.vstack((X1,X2,X3))))

    return Y, X

def stripe2():
    Y1 = sp.vstack((sp.ones((50,1)), sp.zeros((50,1))))
    Y2 = sp.vstack((sp.zeros((50,1)), sp.ones((50,1))))
    Y = sp.hstack([Y1, Y2])

    X1 = sp.random.multivariate_normal([-2,2], [[1,.8],[.8,1]],size=50)
    X2 = sp.random.multivariate_normal([2,-1], [[1,.8],[.8,1]], size=50)
    X = sp.hstack((sp.ones((100,1)),sp.vstack([X1,X2])))

    return Y, X

def betahat(X,Y):
    return la.inv(X.T.dot(X)).dot(X.T).dot(Y)

def mashup_plot(Xdat, Ydat, bounds=[-5,5],res=.1):
    bh = betahat(Xdat, Ydat)
    rang = sp.arange(bounds[0],bounds[1]+res, res)
    X, Y = sp.meshgrid(rang, rang)
    
    Xm = sp.c_[sp.ones_like(X.flatten()), X.flatten(), Y.flatten()]
    Xmdot = Xm.dot(bh)
    
    types = Xmdot.argmax(axis=1)
    print Xm.shape
    #plot the regions
    c = ['b','r','g']
    for i in range(Xmdot.shape[1]):
        plot_on = types==i
        plt.plot(Xm[plot_on,1], Xm[plot_on,2], c[i]+'.', Xdat[Ydat[:,i]==1,1], Xdat[Ydat[:,i]==1,2], c[i]+'o')

    #plot the data segmented
    #tmp = sp.where(Ydat[:,0]==True)[0]
    #plt.plot(Xdat[tmp,1], Xdat[tmp,2], 'o', Xdat[~tmp,1], Xdat[~tmp,2], 'o')
    plt.show()

def steepest_descent(X,Y, step=.001, tol=1e-5, maxiter=5000, bounds=[-5,5], res=.1):
    w = betahat(X,Y)
    a = sp.exp(X.dot(w))
    yhat = (a.T/sp.sum(a, axis=1)).T
    grad = X.T.dot(yhat-Y)
    while la.norm(grad)>tol and maxiter > 0:
        w = w - grad*step
        a=sp.exp(X.dot(w))
        yhat = (a.T/sp.sum(a,axis=1)).T
        grad = X.T.dot(yhat-Y)
        maxiter -= 1

    rang = sp.arange(bounds[0],bounds[1]+res, res)
    Xg, Yg = sp.meshgrid(rang, rang)
    
    Xm = sp.c_[sp.ones_like(Xg.flatten()), Xg.flatten(), Yg.flatten()]
    Xmdot = Xm.dot(w)
    
    types = Xmdot.argmax(axis=1)
    print Xm.shape
    #plot the regions
    c = ['b','r','g']
    for i in range(Xmdot.shape[1]):
        plot_on = types==i
        plt.plot(Xm[plot_on,1], Xm[plot_on,2], c[i]+'.', X[Y[:,i]==1,1], X[Y[:,i]==1,2], c[i]+'o')

    #plot the data segmented
    #tmp = sp.where(Ydat[:,0]==True)[0]
    #plt.plot(Xdat[tmp,1], Xdat[tmp,2], 'o', Xdat[~tmp,1], Xdat[~tmp,2], 'o')
    plt.show()

#---------------------------------------------------------------
#Bayesian Classfier
#---------------------------------------------------------------

def genData(n=1000):
    output = []
    ru = sp.random.uniform
    for i in xrange(n):
        #new person
        person = []
        num = sp.random.uniform()

        #determine driver type
        if num <= .5:
            person.append('C')
            
            num = ru()
            #determine sex
            if num <= .6:
                person.append('F')
            else:
                person.append('M')

            #determine origin
            num = ru()
            if num < .9:
                person.append('LC')
            elif num >= .9:
                person.append('SC')
            else:
                person.append('T')

        elif num >= .8:
            person.append('A')

            num = ru()
            if num <= .8:
                person.append('M')
            else:
                person.append('F')

            num = ru()
            if num <= .5:
                person.append('LC')
            elif num >= .8:
                person.append('T')
            else:
                person.append('SC')
                
        else:
            person.append('B')

            num = ru()
            if num <= .6:
                person.append('M')
            else:
                person.append('F')
                
            num = ru()
            if num <=.8:
                person.append('LC')
            elif num >= .9:
                person.append('SC')
            else:
                person.append('T')
                
        output.append(person)
    return sp.array(output)

def est_prob(data, val):
    """Calculate the probability of P(X)"""

    if val in ['M', 'F']:
        col = 1
    elif val in ['LC', 'SC', 'T']:
        col = 2
    elif val in ['A', 'B', 'C']:
        col = 0

    totalval = sp.where(data[:,col]==val)[0].size
    Total = sp.size(data, axis=0)
    return totalval/float(Total)

def est_condprob1(data, val, given):
    """Calculate the conditional probability of P(X|Y)"""

    if given in ['M', 'F']:
        gcol = 1
    elif given in ['LC', 'SC', 'T']:
        gcol = 2
    elif given in ['A', 'B', 'C']:
        gcol = 0

    if val in ['M', 'F']:
        vcol = 1
    elif val in ['LC', 'SC', 'T']:
        vcol = 2
    elif val in ['A', 'B', 'C']:
        vcol = 0
        
    totalgiven = sp.where(data[:,gcol]==given)[0]
    totalval = sp.where(data[totalgiven][:,vcol]==val)[0]

    return float(totalval.size)/totalgiven.size

def est_condprob2(data, val, given):
    """Calculate the probability of P(X|Y,Z)

    est_condprob2(data, 'A', ['M', 'LC'])"""

    if not isinstance(given, list):
        raise IndexError("Given must be a list or tuple of givens")
    elif len(given) != 2:
        raise IndexError("I need multiple givens!  Give me more...give me more!")

    gcols = []
    for g in given:
        if g in ['M', 'F']:
            gcols.append(1)
        elif g in ['LC', 'SC', 'T']:
            gcols.append(2)
        elif g in ['A', 'B', 'C']:
            gcols.append(0)

    if val in ['M', 'F']:
        vcol = 1
    elif val in ['LC', 'SC', 'T']:
        vcol = 2
    elif val in ['A', 'B', 'C']:
        vcol = 0

    datsize = data.shape[0]
    needed = [val, given[0], given[1]]
    t = sp.where([sp.all(data[i]==needed) for i in range(datsize)])[0]

    t2 = sp.where([sp.all(data[i,1:]==given) for i in range(datsize)])[0]
    
    return float(t.size)/t2.size

def dothedata():
    #generate data
    dat = genData(1000)

    types = ['A', 'B', 'C']
    other = ['M', 'F', 'LC', 'SC', 'T']

    #find P(X)
    for var in types+other:
        print "P(%s) = %f" % (var,est_prob(dat, var))

    #find P(X|Y)
    for t in types:
        for given in other:
            print "P(%s|%s) = %f" % (t, given, est_condprob1(dat, t, given))

    #find P(X|Y,Z)
    for t in types:
        for g1 in ['M', 'F']:
            for g2 in ['LC', 'SC', 'T']:
                print "P(%s|%s,%s) = %f" % (t, g1, g2, est_condprob2(dat,t,[g1,g2]))
