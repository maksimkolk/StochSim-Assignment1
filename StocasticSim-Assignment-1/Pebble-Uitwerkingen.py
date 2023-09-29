import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import asmatrix, asarray, ones, identity, zeros, transpose
from numpy.linalg import eig, inv
from numpy import random
import pytest as pytest
import time
rng = random.default_rng()

### Exercise 1

def drawGraph(A): 
    """Visualizes a graph, given adjacency matrix A""" 
    G = nx.Graph()
    A = asarray(A)
    for row in range(len(A)):
        assert (len(A[row]) == len(A)) , "This is not a square matrix"
        for col in range(row+1, len(A)): #We start at row+1 since we only need to iterate over the upper triangle (without the diagonal)
            if A[row][col] == 1:
                G.add_edge(row+1, col+1)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

drawGraph([[0,1,1],[1,0,1],[1,1,0]]) #draws triangle

#%%
### Exercise 2

def histOfEigenvalues(A, d, p_d):
    """Gives a histogram of the eigenvalues of the centered matrix of A"""
    assert len(A) == d, "Length of matrix A must equal d" 
    A = np.asmatrix(A)
    J_d = np.asmatrix(np.ones((d,d)))
    I_d = np.asmatrix(np.identity(d))
    centered = A - p_d*(J_d-I_d)
    eigenvalues = np.linalg.eig(centered)[0]
    nr_bins = max(int(round((max(eigenvalues)) - min(eigenvalues))), 8)
    plt.hist(eigenvalues, bins=nr_bins)
    plt.title(f'Eigenvalues of centered Erdös-Rényi adjacency matrix with $d={d}$ and $p(d)={p_d}$', fontsize = 20, weight = 'bold')
    plt.ylabel('count', fontsize = 18)
    plt.xlabel('eigenvalues', fontsize = 18)
    plt.show()
    return eigenvalues

    

histOfEigenvalues([[0,1,1],[1,0,1],[1,1,0]], 3, 1) #Ouputs 3 times the eigenvalue 0

#%%
### Exercise 3

def ErdosRenyi(p_d, d, nrErdos):
    """Generates nrErdos d x d Erdös-Rényi matrices A with bernoulli parameter p_d"""
    
    A = rng.choice([0, 1], p=[(1-p_d), p_d], size=(nrErdos, d, d))

    diag_ind = [i for i in range(d)]

    A[:, diag_ind, diag_ind] = 0
    A = A - np.tril(A)
    triuA = np.triu(A)
    for i in range(nrErdos):
	    triuA[i] = triuA[i].T
    A = A+triuA

    return A


def test_ErdosRenyi():
    assert (ErdosRenyi(0, 7, 1) == np.zeros((7,7))).all(), 'If p_d = 0, ErdosRenyi() should output an all zero matrix'
    assert (ErdosRenyi(1, 7, 1) == np.ones((1,1)) - np.identity(7)).all(), 'If p_d = 1, ErdosRenyi() should output an all one matrix'

test_ErdosRenyi()

def simulateSn(p_d, n, d):
    """Generate S_n = sum^n_i=1 X_i, where X_i is an d x d Erdös-Rényi matrix with bernoulli parameter p_d"""
    return np.sum(ErdosRenyi(p_d, d, n), axis=0)

def expectedSn(trials, p_d, nrErdos, d):
    """Returns the expected value of S_n, based on the number trials"""

    B = rng.choice([0, 1], p=[(1-p_d), p_d], size=(trials, nrErdos, d, d))

    diag_ind = [i for i in range(d)]
    
    for x in range(trials):
        B[x][:, diag_ind, diag_ind] = 0
        B[x] = B[x] - np.tril(B[x])
        triuB = np.triu(B[x])
        for i in range(nrErdos):
            triuB[i] = triuB[i].T
        B[x] = B[x]+triuB

    Sn = np.sum(B, axis=1)

    return np.mean(Sn, axis=0)

def test_Sn():
    assert (simulateSn(1, 5, 3) == (ones((3,3)) - identity(3))* 5).all(), "simulateSn(1,5,3) should output a 3 x 3 matrix with zeros on diagonal and 5's everywhere else"
    assert (expectedSn(20, 1, 5, 3) == (ones((3,3)) - identity(3))* 5).all() , "expectedSn(20,1,5,3) should output a 3 x 3 matrix with zeros on diagonal and 5's everywhere else"

test_Sn()

# Verifying output exercise 3

def test_verifyExpectedSn(p_d, n, d):
    """Test the relationship mu_n/n equals p_d(J_d - I_d)"""
    mu_n = expectedSn(1000, p_d, n, d)
    J_d = asmatrix(ones((d,d)))
    I_d = asmatrix(identity(d)) 
    print(f'We should have {p_d*(J_d-I_d)}, and we have { mu_n / n}')
    
test_verifyExpectedSn(0.5, 5, 4)

### Exercise 4

def SimulateV(n, Z, mu, p_d, d):
    """Simulates V = (S_n-mu)^T Z(S_n-mu)"""
    S_n = asmatrix(simulateSn(p_d, n, d))
    Z = asmatrix(Z)
    return asarray(transpose(S_n - mu)*Z*(S_n - mu))
    
def expectedV(trials, n, Z, mu, p_d, d):
    """Determines expected value of V = (S_n-mu)^T Z(S_n-mu) based on trials"""
    total = zeros((d,d))
    for _ in range(trials):
        total += SimulateV(n, Z, mu, p_d, d)
    total = asarray(total)
    return total / trials

def test_V():
    Z = identity(2)
    trials = 30
    p_d=1
    d =2 
    n =3
    assert (expectedV(trials, n, Z, expectedSn(trials, p_d, n, d), p_d, d) == zeros((2,2))).all(),"For p_d = 1, d =2 and n=3, V =[[0,0],[0,0]]"

test_V()

#%%
### Exercise 5

def simulateL(Z, h, mu, ni):
    """Simulates lambda_max * (Z^{-1} + h*mu +vi)"""
    A = inv(Z) + h*mu + ni
    lambda_max = max(eig(A)[0])
    return lambda_max

# The function expectedL is extra, and was not required, however it is used to test the function simulateL

def expectedL(trials, n, Z, h, mu, ni, p_d, d):
    """Calculates expected value of lambda_max * (Z^{-1} + h*mu +vi) based on trials"""
    total = zeros((d,d))
    for _ in range(trials):
        total += simulateL(Z, h, mu, ni)
    return total/trials

def test_L():
    Z = identity(2)
    trials = 30
    p_d = 1
    d = 2 
    n = 3
    h = 1
    mu = expectedSn(trials, p_d, n, d)
    assert (expectedL(trials, n, Z, h, mu, expectedV(trials, n, Z, mu, p_d, d), p_d, d) == pytest.approx(np.zeros((2,2))+9*np.ones((2,2))-6*np.identity(2))),"For p_d = 1, d = 2 and n = 3, L =[[3,9],[9,3]]"

test_L()

#%%
### Exercise 6

def generateZ(sigma, d):
    """Generate a positive semi definite matrix Z"""
    assert len(sigma) == d, "Sigma should be a d x d matrix"
    Z = zeros((d,d))
    for _ in range(d):
        G = np.asmatrix(rng.multivariate_normal(np.zeros(d), sigma))
        Z += np.transpose(G)*G   #Note that sampling from the multivariate_normal gives a row vector, therefore we switch the transpose multiplication
    return Z

def test_generateZ():
    sigma = identity(3)
    d = 3
    Z = generateZ(sigma,d)
    assert (eig(Z)[0] > 0).all(), 'not all eigenvalues are positive'

test_generateZ()

#%%
### Exercise 7

# Options to implement:
# - generate random matrix, and multiply by its transpose, since a matrix A*A^T is always positive definite
#   Note: Not all possible matrices can be generated this way.

def generateZ_V2(d):
    """Generate a positive definite matrix Z"""
    Z = np.zeros((d,d))
    for row in range(d):
        for col in range(d):
            bit = 0
            while bit == 0:
                bit = rng.uniform(0,1)
            Z[row][col] = (1/bit) #Every entry is an positive float (all 
            #positive definite matrices can be generated, but not with equal distribution)
    Z = np.asmatrix(Z)
    return Z*np.transpose(Z)

def test_generateZ_V2():
    Z = generateZ_V2(5)
    assert (eig(Z)[0] > 0).all(), 'not all eigenvalues are positive'

test_generateZ_V2()
#print(generateZ_V2(5))

#%%
##################################
### DELIVERABLE
##################################

# Program 1:

def visualizeEigValErdosRenyi(d,p):
    A = ErdosRenyi(p,d,1)
    eigenvalues = np.linalg.eig(A)[0]
    plt.hist(eigenvalues)
    plt.show()
    return A, eigenvalues

visualizeEigValErdosRenyi(3,1) 

# Program 2:    (NOT TESTED YET)

def estimateSFree(d, m, n, p, sigma, trials,mu):
    """Estimates an upperbound of ||S^n||_free based on the function generateZ()"""
    #trials = 100 # used to determine the number of trials to compute mu and ni
    #mu = expectedSn(trials,p,n,d) 
    minimum1 = 'no number assigned' # minimum when using h = 1
    minimum2 = 'no number assigned' # minimum when using h = -1
    
    for _ in range(m): 
        h = 1
        Z = generateZ(sigma, d)
        ni = expectedV(trials,n,Z,mu,p,d)
        L = simulateL(Z,h,mu,ni) # creating L with h =1
        if minimum1 == 'no number assigned' or L < minimum1:
            minimum1 = L
        h = -1
        L = simulateL(Z,h,mu,ni)
        if minimum2 == 'no number assigned' or L < minimum2:
            minimum2 = L
    return max(minimum1, minimum2)


# Program 3:    (NOT TESTED YET)

def estimateSFree_V2(d, m, n, p, trials,mu):
    """Estimates an upperbound of ||S^n||_free based on the function generateZ()"""
    #trials = 100 # used to determine the number of trials to compute mu and ni
    #mu = expectedSn(trials,p,n,d) 
    minimum1 = 'no number assigned' # minimum when using h = 1
    minimum2 = 'no number assigned' # minimum when using h = -1
 
    for _ in range(m): 
        h = 1
        Z = generateZ_V2(d)
        ni = expectedV(trials,n,Z,mu,p,d)
        L = simulateL(Z,h,mu,ni)
        if minimum1 == 'no number assigned' or L < minimum1:
            minimum1 = L
        h = -1
        L = simulateL(Z,h,mu,ni)
        if minimum2 == 'no number assigned' or L < minimum2:
            minimum2 = L
    return max(minimum1, minimum2)

### Exercise a)

A = ErdosRenyi(0.3, 20, 1)
B = ErdosRenyi(0.3, 20, 1)
C = ErdosRenyi(0.3, 20, 1)
# drawGraph(C)

### Exercise b)

#for d in [100, 1000]:
#    for p in [0.2, 0.5, 0.8]:
#        A = ErdosRenyi(p, d, nrErdos)
#        eigenvalues = histOfEigenvalues(A, d, p)

#%%
### Exercise c) 
# given values
sigma = identity(10)
d = 10
trials = 200
p = 0.7
n = 1

# start experiment ex 6
#measure time
start_wishart = time.time()

lambda1 = [] # h = 1
lambda2 = [] # h = -1
mu = expectedSn(trials,p,n,d) 
test_length = 10000
for m in range(test_length):
    Z = generateZ(sigma, d)
    ni = expectedV(trials,n,Z,mu,p,d)
    h = 1
    lambda1.append(simulateL(Z,h,mu,ni))
    h = -1
    lambda2.append(simulateL(Z,h,mu,ni))


values = [max((min(lambda1[:i+1]), min(lambda2[:i+1]))) for i in range(test_length)]
values

plt.plot(range(1, test_length +1), values)
plt.xlabel('$m$')
plt.ylabel('Upperbound for $\|S_n\|_{free}$')
plt.show()

end_wishart = time.time()
runtime_wishart = end_wishart-start_wishart
print("The runtime of deliverable c with PD matrices created with Wishart is ", runtime_wishart)

#%%
# start experiment ex 7
#measure runtime
start_transpose = time.time()

lambda1 = [] # h = 1
lambda2 = [] # h = -1
mu = expectedSn(trials,p,n,d) 
test_length = 10000
for m in range(test_length):
    Z = generateZ_V2(d)
    ni = expectedV(trials,n,Z,mu,p,d)
    h = 1
    lambda1.append(simulateL(Z,h,mu,ni))
    h = -1
    lambda2.append(simulateL(Z,h,mu,ni))


values = [max((min(lambda1[:i+1]), min(lambda2[:i+1]))) for i in range(test_length)]
values

plt.plot(range(1, test_length +1), values)
plt.xlabel('$m$')
plt.ylabel('Upperbound for $\|S_n\|_{free}$')
plt.show()

end_transpose = time.time()
runtime_transpose = end_transpose-start_transpose
print("The runtime of deliverable c with PD matrices created with transposes is ", runtime_transpose)

#%%
### Exercise d
tableMean = np.zeros((3,19))
tableHalftime = np.zeros((3,19))
d, m, sigma = 10, 10, np.identity(10)

trials = 50
nrRuns = 100
for i in range(0,3):
    for j in range(0,19):
        n = 10**i       # Compute corresponding (n,p) for entry (i,j) of the table
        p = (j+1)*0.05
        mu = expectedSn(200,p,n,d) 
        print('calculated mu')
        sumUpperbound = 0  #Sum of uppebounds
        sum2Upperbound = 0 #Sum of upperbounds squared
        for _ in range(nrRuns): #Determine the mean and variance of the upperbound
            estimate = estimateSFree(d, m, n, p, sigma, trials, mu)
            sumUpperbound += estimate
            sum2Upperbound += estimate**2
            print(_)
        meanUpperbound = sumUpperbound/nrRuns
        varianceUpperbound = sum2Upperbound/nrRuns - meanUpperbound**2
        halfwidth = 1.96*np.sqrt(varianceUpperbound/nrRuns)
        tableMean[i][j] =  meanUpperbound
        tableHalftime[i][j] = halfwidth
        print(f'(n,p)={(n,p)}, {meanUpperbound} +- {halfwidth}')

tableMean
#d, m, sigma = 10, 10, identity(10)




#n =5
#p = 0.5
#estimateSFree(d, m, n, p, sigma,10)
#expectedSn(100,p,n,d) 