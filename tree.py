#Qualitative Features: Class, Sex
#Quantitative Features: Age
#Y = survival

import scipy as sp
import itertools

class Node(object):
    r'''
    Node class for tree

    Expected properties:
        left (type: Node):  Node on left side
        right (type: Node): Node on right side

        x (nxd): input vectors that have reached this part of the tree
        y (nx1): corresponding classes
        qual (1xd): Boolean vector indicating the qualitative features
        level: int indicating depth of subtree
    
    '''
    def __init__(self,x,y,qualmask,level=0):
        self.level=level
        self.x = x
        self.y = y

        self.left = None
        self.right = None
        
        self.qualmask = sp.asarray(qualmask).astype('bool')

        print "level", self.level
        self.split()
        
    def predict(self,part):
        r'''
        Only called on leaf nodes for making predictions
        part: (nx1) Boolean vector suggesting a data partition to send to Left Child
        returns: (Mode of y where part is True,Mode of y where part is False)
        '''
        pass

    def split(self):
        r'''Split the data'''

        features = self.x.shape[1]
        errkeys = ['mask', 'gini', 'split', 'rule', 'feat']
        errvals = [None, None, None, None, None]
        #iterate over each feature
        for f in range(features):
            qx = self.x[:,f]
            lsterr = gini(qx,self.y, qual=self.qualmask[f])
            if lsterr[1]<errvals[1]:
                errvals[:3] = lsterr
                errvals[4] = f
            elif errvals[1] is None:
                errvals[:3] = lsterr
                errvals[4] = f          

            errvals[3]=self.qualmask[f]

        d = dict(zip(errkeys,errvals))
        self.info=d
        print '[%s : %d|%s] gini: %f' % (d['rule'], d['feat'], str(d['split']), d['gini'])
        if self.level < 3 :
            self.left = Node(self.x[d['mask']], self.y[d['mask']], self.qualmask, self.level+1)
            self.right = Node(self.x[~(d['mask'])], self.y[~(d['mask'])], self.qualmask, self.level+1)
        else:
            print "Maximum depth reached!\n"
                
def gini(x,y, qual=False):
    r'''
    INPUTS:
        x: a 1d vector corresponding to the feature to use for calculating gini index
        y: a 1d vector of the classes

    OUTPUTS:
        lsterr_split: tuple of partition corresponding to least error (left node, right node)
        lsterr      : the gini index of the lsterr_split
    '''

    def cgini(L, N, y):
        lTrue = float(L.sum())
        lFalse = float((~L).sum())

        ayl = (y[L]==1).sum()
        byl = (y[L]==0).sum()

        ayr = (y[~L]==1).sum()
        byr = (y[~L]==0).sum()
        
        
        a = ayl*(1-ayl/lTrue) + byl*(1-byl/lTrue)
        b = ayr*(1-ayr/lFalse) + byr*(1-byr/lFalse)

        gini_score = 2*(a+b)/(lTrue+lFalse)

        return gini_score
        

    u = sp.unique(x)
    N = x.size
    #lsterr  [mask, err, split]
    lsterr = [None, 1e6, None]

    if qual==True:                
        complete=set(u)
        power = powerset(u)

        for p in power:
            left = p
            right = tuple(complete.symmetric_difference(p))
            
            L_left = sum(x==val for val in left).astype('bool')

            err=cgini(L_left, N, y)
            if err < lsterr[1]:
                lsterr[1] = err
                lsterr[0] = L_left
                lsterr[2] = p
    else:
        for i in u:
            left = x>i

            err = cgini(left, N, y)
            if err<lsterr[1]:
                lsterr[1]=err
                lsterr[0]=left
                lsterr[2] = i

    return lsterr
    

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)/2+1))
