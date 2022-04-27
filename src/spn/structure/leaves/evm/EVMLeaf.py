"""
@author: Stefan Luedtke
"""

import numpy as np
from numpy import *
#from operator import mul
from operator import *
import scipy
from collections import Counter
import sys, traceback
from scipy import stats
import datetime
from functools import *


import pandas as pd
from scipy.special import binom
from scipy.stats import chisquare
from collections import namedtuple

from itertools import product

from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.structure.leaves.parametric.MLE import update_parametric_parameters_mle



from spn.structure.leaves.parametric.Parametric import Leaf
class EVM(Leaf):
    def __init__(self, comp, latentProb, scope=None):
        Leaf.__init__(self, scope=scope)
        self.comp = comp
        self.latentProb = latentProb
        
    property_type = namedtuple("EVM", "comp latentProb")
    
    @property
    def parameters(self):
        return __class__.property_type(comp=self.comp, latentProb = self.latentProb)

def create_evm_leaf(local_data,ds_context,scope,alpha=0,numComponents = 3):
    if len(scope) == 1:
        #print("create Bernoulli")
        node = Bernoulli()
        node.scope.extend(scope)
        update_parametric_parameters_mle(node, local_data,alpha)
        return node
    else:
        comp, latentProb = em_evm(local_data,numComponents = numComponents)
        node = EVM(comp = comp, latentProb = latentProb)
        node.scope.extend(scope)
        return node


# Code below taken from Niepert's implementation of Exchangeable Variable Models


# computes the binomial coefficient
def  n_take_k(n,r):
  
    if r > n-r:  # for smaller intermediate values
        r = n-r
    return int( reduce( mul, range((n-r+1), n+1), 1) /
      reduce( mul, range(1,r+1), 1) )


# class representing one mixture component
class MComponent:
	size = 1
	normalize = 1.0
	nk = dict()
	sampleWeights = dict()
	laplace = 0.1

	# part is the partition in form of an array [0, 0, 1, 0, 1, 2, 3, 1] indicating column membership on (here: 4) partitions
	def __init__(self, data, weights):
		
		# get dimensions of the data matrix
		m,n = data.shape

		# stores the possible binomial coefficients (caching)
		self.nk = dict()
		# stores the accumulated weights 
		self.sampleWeights = dict()
		# size of the block
		self.size = n

		# go through all rows of the data table and store number of configurations
		for i in arange(m):
			configs = count_nonzero(data[i])
			#print configTuple
			# configs[i] stores the count for each block
			sWeight = self.sampleWeights.get(configs, -1.0)
			#print sWeight
			if sWeight > 0:
				sWeight += weights[i]
				#print sWeight
				self.sampleWeights[configs] = sWeight
			else:
				self.sampleWeights[configs] = weights[i]

		# number of configurations (= number of ones) that have no occurence in the training data
		diff = n - len(self.sampleWeights) + 1.0

		#  the normalization constant for the probabilities of the block configurations
		self.normalize = float(sum(list(self.sampleWeights.values())))

		# perform Laplacian smoothing -> add laplace constant to each possible configuration
		self.normalize += diff*self.laplace
		for i in list(self.sampleWeights):
			self.sampleWeights[i] += self.laplace
			self.normalize += self.laplace

		#testSum = 0.0
		#for i in arange(n+1):
		#	sw = self.sampleWeights.get(i, -1.0)		
		#	if  sw > 0.0:
		#		testSum += sw
		#	else:
		#		testSum += self.laplace
		#
		#print "testSum: ",testSum
		#print "smooth: ",self.smooth

	# returns the probability of one particular configuration (here: conditional probability)
	def prob(self, data_point):

		# count the number of ones of the given example
		configs_test = count_nonzero(data_point)

		# look up the probability of the given block configuration
		sw = self.sampleWeights.get(configs_test, -1.0)		
		if  sw > 0.0:
			# return the normalized probability
			currProb = float(sw)/self.normalize
		else:
			# return a Laplace smoothed probability
			currProb = self.laplace/self.normalize

		# normalize by the number of configuration represented by this particular block configuration
		# we cache prviously computed binomial coefficients to make this more efficient
		nvalue = self.size
		kvalue = configs_test
		cnk = tuple([nvalue, kvalue])
		tst = self.nk.get(cnk, False)
		if tst:
			currProb = currProb / tst
		else:
			tst = n_take_k(nvalue, kvalue)
			self.nk[cnk] = tst
			currProb = currProb / tst

		return currProb

	# returns the log-probability of one particular configuration (here: conditional probability)
	def probLog(self, data_point):

		# count the number of ones of the given example
		configs_test = count_nonzero(data_point)

		# look up the probability of the given block configuration
		sw = self.sampleWeights.get(configs_test, -1.0)		
		if  sw > 0.0:
			# return the normalized probability
			currProb = float(sw)/self.normalize
		else:
			# return a Laplace smoothed probability
			currProb = self.laplace/self.normalize

		# normalize by the number of configuration represented by this particular block configuration
		# we cache prviously computed binomial coefficients to make this more efficient
		nvalue = self.size
		kvalue = configs_test
		cnk = tuple([nvalue, kvalue])
		tst = self.nk.get(cnk, False)
		if tst:
			currProb = log(currProb) - log(float(tst))
		else:
			tst = n_take_k(nvalue, kvalue)
			self.nk[cnk] = tst
			currProb = log(currProb) - log(float(tst))

		return currProb


class IndComponent:

	comp = array([])
	splitNumbers = [0]
	partition = array([])
	assignment = array([])

	# data is the data matrix, weights is the weight matrix (fractional assignment weights) of the EM algorithm
	def __init__(self, data, weights):

		# get the dimensions of the trainging data matrix
		m,n = data.shape

		#############################################################################
		########## the following lines of code compute the block structure ##########
		#############################################################################
		# this array contains the data points multiplied by the weights
		scaledData = zeros((m,n),dtype=float)

		# scale the data matrix with the weights of the examples
		for i in arange(m):
			scaledData[i] = data[i] * weights[i]

		# compute the means of the scaled data values
		msTemp = zeros(n,dtype=float)
		for i in arange(n):
			msTemp[i] = mean(scaledData[:,i])

		# represents the assignment to blocks
		assign = zeros(n, dtype=int)
		# counter keeping track of the number of blocks
		countUnique = 0

		# sorted the marginals and keep track of the order
		sortedMsTempArg = argsort(msTemp)
		sortedMsTemp = sort(msTemp)

		# the current value for comparison
		previousValue = sortedMsTemp[0]

		# keeps track of the element with the smallest marginal in the current block
		smallestInBlock = 0

		# iterate through the unique marginals
		for j in arange(n):
			# only if there is a difference in means do we have to run a t-test
			if abs(previousValue-sortedMsTemp[j]) > 0.0:
					# compute the p-value of the Welch's t-test
					p = stats.ttest_ind(scaledData[:,sortedMsTempArg[j]], scaledData[:,sortedMsTempArg[smallestInBlock]], equal_var=False)[1]
					# if p-value strictly smaller than 0.1 seperate the variables (and, therefore, blocks)
					if  p < 0.1:
						countUnique += 1
						smallestInBlock = j
			assign[sortedMsTempArg[j]] = countUnique
			previousValue = sortedMsTemp[j]

		# the integers used to index the blocks
		self.splitNumbers = unique(assign)
		# copy the partition indicator array to the class variable "partition"
		self.partition = assign

		# initialize array of exchangeable components
		self.comp = array([])
		# stores the block structure so that we don't have to recompute it every time
		self.assignment = zeros((len(self.splitNumbers), n), dtype=bool)

		# create independent components based on the block structure
		for i in self.splitNumbers:
			self.comp = append(self.comp, MComponent(data[:,self.partition==i], weights))
			self.assignment[i] = array([self.partition==i])
	

	# simply update the parameters not the structure
	def update(self, data, weights):
		self.comp = array([])
		for i in self.splitNumbers:
			self.comp = append(self.comp, MComponent(data[:,self.assignment[i]], weights))

	# return probability of given example
	def prob(self, data_point):
		# iterate over the number of blocks
		pr = 1.0
		for i in arange(len(self.comp)):
			pr = pr * self.comp[i].prob(data_point[self.assignment[i]])
				
		return pr

	# compute log-probability of given example
	def probLog(self, data_point):
		
		# iterate over the number of blocks
		pr = 0.0
		for i in arange(len(self.comp)):
			pr = pr + self.comp[i].probLog(data_point[self.assignment[i]])

		return pr

	# return the number of blocks
	def getNumberOfBlocks(self):
		return float(len(self.comp))
    


def em_evm(data,numComponents=20):
    
    # get the dimensions of the trainging data matrix
    m,n = data.shape

    # the number of mixture components (latent variable values)
    #numComponents = 20

    # generate random matrix
    initData = np.random.randint(2, size=(m, n))

    # split the random matrix. this is used to initialize EM
    initDataIndicator = np.random.randint(numComponents, size=(m,))

    # comp are the mixture components
    comp = array([])
    for j in arange(numComponents):
        initAssign = zeros(m, dtype=float)
        initAssign[initDataIndicator==j] = 1.0
        # create a mixture component for the ith row having value '0'
        compTemp = IndComponent(initData, initAssign)
        # append to the list of mixture components
        comp = append(comp, compTemp)

    # class probabilities initialized to uniform probabilities
    latentProb = ones(numComponents, dtype=float)
    latentProb = latentProb/sum(latentProb)

    #print("latent class probability: ",latentProb)

    averageLL = 0.0

    for c in arange(100):
        #print("EM iteration: ",c)
        # iterate over the training samples (all of them) an compute probability
        compPr = zeros(numComponents, dtype=float)
        weights = zeros((numComponents, m), dtype=float)
        # the E step
        for i in arange(m):
            probSum = 0.0
            for j in arange(numComponents):
                # probability (unnormalized) of the data point i for the component j
                if latentProb[j] > 0.0:
                    # looks weird but is numerically more stable
                    prob = exp(log(latentProb[j]) + comp[j].probLog(data[i]))
                else:
                    prob = 0.0

                weights[j][i] = prob
                # the sum of the probabilites (used for normalization)
                probSum += prob
                #print weights[j][i]
        
            for j in arange(numComponents):
                # normalize the probabilities
                if probSum <= 0.0:
                    weights[j][i] = 0.0
                else:
                    weights[j][i] = weights[j][i] / probSum
                # aggregate the normalized probabilities
                compPr[j] += weights[j][i]
                #print weights[j][i],

        # the M step
        # update the class priors
        latentProb = compPr/sum(compPr)

        # update the parameters of the mixture components
        # comp are the mixture components
        comp2 = array([])

        # keep track of the statistics of the block sizes
        blockStatistics = zeros(numComponents, dtype=float)

        # run inference in the components and compute the new probabilities
        for j in arange(numComponents):
            # create the j-th mixture component 
            compTemp = IndComponent(data, weights[j])
            #add the mixture component to the new structure
            comp2 = append(comp2, compTemp)
            #update the j-th mixture component of the previous structure
            comp[j].update(data, weights[j])

        # compute the log-likelihood on the training data for both candidate structures
        # and chose the one with the highest log-likelihood score
        sumn1 = 0.0
        sumn2 = 0.0
        for x in data:
            prob1 = 0.0
            prob2 = 0.0
            for j in arange(numComponents):
                if latentProb[j] > 0.0:
                    prob1 = prob1 + exp(log(latentProb[j]) + comp[j].probLog(x))
                    prob2 = prob2 + exp(log(latentProb[j]) + comp2[j].probLog(x))
            sumn1 += log(prob1)
            sumn2 += log(prob2)

        # if the new structure has higher log-likelihood, choose it
        if sumn2 >= sumn1:
            comp = comp2
            averageLLNew = sumn2/float(m)
        else:
            averageLLNew = sumn1/float(m)
            #print "The previous structure was chosen."

        # compute the block sizes; this is for computing the mean and std dev of the number of blocks
        for j in arange(numComponents):
            blockStatistics[j] = comp[j].getNumberOfBlocks()
        #print("blocks;  mean: ", mean(blockStatistics), "; stddev: ", std(blockStatistics))

        #print("Current average log-likelihood on the training data: ",averageLLNew)
        #print("Difference in average log-likelihood: ",abs(averageLLNew - averageLL))

        # stop EM when probs are not changing anymore
        if abs(averageLL - averageLLNew) < 0.001:
            break

        # set the old ll to the current one
        averageLL = averageLLNew
        
    return (comp,latentProb)




def evm_likelihood(node,data=None,dtype=np.float64):
    dd = data[:,node.scope]
    
    comp = node.comp
    latentProb = node.latentProb
    # dimensions of test data
    mt,nt = dd.shape
    numComponents = len(comp)

    # compute the log-likelihood of the test data for the partial exchangeable model
    prob = np.zeros(dd.shape[0])
    for i in range(dd.shape[0]):
        x = dd[i,:]
        for j in arange(numComponents):
            if latentProb[j] > 0.0:
                prob[i] = prob[i] + exp(log(latentProb[j]) + comp[j].probLog(x))
    prob = prob.reshape(prob.shape[0],1)
    return prob


from spn.algorithms.Inference import add_node_likelihood
add_node_likelihood(EVM, evm_likelihood)
