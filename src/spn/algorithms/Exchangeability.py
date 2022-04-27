"""
@author: Stefan Luedtke
"""


import pandas as pd
from scipy.special import binom
from scipy.stats import chisquare
from scipy import stats
import numpy as np
from collections import namedtuple

from itertools import product

from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe

from spn.algorithms.Inference import log_likelihood, likelihood


def create_exchangeable_leaf(local_data,ds_context,scope,alpha=0):
    #idx = scope[0]
    params = getExchParams(local_data,alpha) 
    
    node = Exchangeable(params = params)
    node.scope.extend(scope)

    return node


def getExchParams(data,alpha=0):

    #given that data is fully exchangeable, learn the parameters
    #simply by maximum likelihood: count conversion, then relative frequency
    # + laplace smoothing 
    dd = pd.DataFrame(data)
    sums = dd.sum(1)
    #make array that has length of all possible sums
    params = np.zeros(data.shape[1]+1)
    for i in range(len(params)):
        params[i] = np.sum(sums == i) + (alpha * binom(len(params)-1,i) / 2**data.shape[1])
    norm = np.sum(params)
    params = np.array([params[i] /norm / binom(len(params)-1,i) for i in range(len(params))])
    return pd.DataFrame(params)

from spn.structure.leaves.parametric.Parametric import Leaf
class Exchangeable(Leaf):
    def __init__(self, params, scope=None):
        Leaf.__init__(self, scope=scope)
        self.params = params
        
    property_type = namedtuple("Exchangeable", "params")
    
    @property
    def parameters(self):
        return __class__.property_type(params=self.params)

def exchangeable_likelihood(node,data=None,dtype=np.float64):
    dd = data[:,node.scope]
    params = node.params
    #for each row in data: count number of ones, then read off probability from params 
    nones = dd.sum(axis=1).astype(int)
    probs =params.loc[nones].copy()
    probs[np.isnan(probs)]  = 0 #capture the case where the count does not appear in params
    return probs.values #.reshape(data.shape[0])


from spn.algorithms.Inference import add_node_likelihood
add_node_likelihood(Exchangeable, exchangeable_likelihood)

from spn.io.Text import add_node_to_str
add_node_to_str(Exchangeable,lambda node, x, y: "Exchangeable("+str(list(node.params.loc[:,0]))+"|"+str(node.scope)+")")


#inspired by Niepert's test for full exchangeability:
#sweep once through RVs, and test whether they have same mean (and variance)
#Niepert used Welch's t test, but as we have Bernoulli RVs, we use
#-> 2-sample z-test aka prop.test in R. 
def isExchangeable_viaExpectation(data,significance=0.05):
    colsums = np.sum(data,0)
    table = np.array([np.array([xi,data.shape[0]]) for xi in colsums])
    z,p,_,_ = stats.chi2_contingency(table)
    return p>significance

# our proposed exchangeability test: Chi squared test:
# for a set of RVs, compute probabilities of each assignment 
# that would arise when distribution would be exchangeable,
# then do goodness-of-fit test with actual data, using chi squared test
def isExchangeable_viaChiSquared(data,significance=0.05):
    exparams = getExchParams(data,alpha=0)
    exparams.loc[:,exparams.shape[1]] = np.array(exparams.index).astype(int)
    dd = pd.DataFrame(data)
    assignm = list(list(product(* data.shape[1]*[[0,1]])))
    fullparams = np.zeros(len(assignm))
    if(len(assignm[0])==2):
        for j in range(len(assignm)):
        #TODO: ONLY WORKS FOR 2 COLUMNS OF DATA...
            a,b = assignm[j]
            fullparams[j] = np.sum(np.logical_and(data[:,0] == a,data[:,1] == b))
    else:
        for j in range(len(assignm)):
            v = np.apply_along_axis(lambda x: np.all(x == list(assignm[j])),1,data)
            fullparams[j] = np.sum(v)
            
    fullparams = pd.DataFrame(fullparams)
    fullparams.loc[:,1] = [sum(ass) for ass in assignm]
    fullparams.columns=["freq","nones"]
    exparams.columns = ["prob","nones"]
    ff = pd.merge(exparams,fullparams,on="nones")
    #now the statistical test: 
    p = chisquare(ff.loc[:,"freq"],ff.loc[:,"prob"]*data.shape[0]).pvalue
    return p>significance

#the full chisquared test is infeasible for many RVs. thus, perform pairwise and assume
# exchangeability when all RVs are exchangeable with the first one
def isExchangeable_viaChiSquared_pairwise(data,significance=0.05):
    nRV = data.shape[1]
    for i in range(1,nRV):
        if not isExchangeable_viaChiSquared(data[:,[0,i]],significance=significance):
            return False
    return True

#same as above, but only some proportion of all tests need to return true
def isExchangeable_viaChiSquared_pairwise_proportion(data,significance=0.05,proportion=0.7):
    nRV = data.shape[1]
    isEx = [isExchangeable_viaChiSquared(data[:,[0,i]],significance=significance) for i in range(1,nRV)]
    pr = np.mean(isEx)
    return pr >= proportion


def ex_bottom_up_log_ll(node, data, dtype=np.float64):
    #basically, following steps:
    #1. get ll on all data rows (that do not have missing data)
    #2. get the most likely values for the missing cells 
    #   (in parametric, this is simply the mode of the node dist, but 
    #   in Ex, can try both values and use the one with higher prob)
    #3. return all probs (i.e. one prob for each row: either from step 1, or the higher)
    
    nan_col = np.isnan(data[0,:])

    if not np.isin(np.where(nan_col)[0][0] ,node.scope):
        return exchangeable_likelihood(node, data)
    
    #all 0
    data[:,nan_col]  = 0
    lls0 = exchangeable_likelihood(node,data)
    #all 1
    data[:,nan_col] = 1
    lls1 = exchangeable_likelihood(node,data)
    #for each row in data: select 0 if lls0 higher, 1 otherwise
    ass = np.where(lls0 > lls1,0,1)
    data[:,nan_col] = ass
    #for each row, select higher ll
    ll = np.where(lls0 > lls1,lls0,lls1)
    return ll


def ex_top_down(node, input_vals, data, lls_per_node=None, dtype=np.float64):
    return


add_node_mpe(Exchangeable, ex_bottom_up_log_ll, ex_top_down)
