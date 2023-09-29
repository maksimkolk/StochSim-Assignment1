import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from tqdm import tqdm
from time import time
import pytest as pytest
rng = random.default_rng()

### Exercise 1

def drawGraph(A): 
    """Visualizes a graph, given adjacency matrix A""" 
    G = nx.Graph()
    A = np.asarray(A)
    for row in range(len(A)):
        assert (len(A[row]) == len(A)) , "This is not a square matrix"
        for col in range(row+1, len(A)): #We start at row+1 since we only need to iterate over the upper triangle (without the diagonal)
            if A[row][col] == 1:
                G.add_edge(row+1, col+1)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

# drawGraph([[0,1,1],[1,0,1],[1,1,0]]) #draws triangle

### Exercise 2

def histOfEigenvalues(A, d, p_d):
    """Gives a histogram of the eigenvalues of the centered matrix of A"""
    assert len(A) == d, "Length of matrix A must equal d" 
    A = np.asmatrix(A)
    J_d = np.asmatrix(np.ones((d,d)))
    I_d = np.asmatrix(np.identity(d))
    centered = A - p_d*(J_d-I_d)
    eigenvalues = np.linalg.eig(centered)[0]
    nr_bins = int(round((max(eigenvalues)) - min(eigenvalues)))
    plt.hist(eigenvalues, bins=nr_bins, edgecolor = 'k')
    #^^^^^^^^ aanpassing
    plt.title(f'Eigenvalues of centered Erdös-Rényi adjacency matrix with d={d} and p(d)={p_d}')
    plt.ylabel('count')
    plt.xlabel('eigenvalues')
    plt.show()
    return eigenvalues
    

# histOfEigenvalues([[0,1,1],[1,0,1],[1,1,0]], 3, 1) #Ouputs 3 times the eigenvalue 0

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

# test_ErdosRenyi()

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
    assert (simulateSn(1, 5, 3) == (np.ones((3,3)) - np.identity(3))* 5).all(), "simulateSn(1,5,3) should output a 3 x 3 matrix with zeros on diagonal and 5's everywhere else"
    assert (expectedSn(20, 1, 5, 3) == (np.ones((3,3)) - np.identity(3))* 5).all() , "expectedSn(20,1,5,3) should output a 3 x 3 matrix with zeros on diagonal and 5's everywhere else"

# test_Sn()

# Verifying output exercise 3

def test_verifyExpectedSn(p_d, n, d):
    """Test the relationship mu_n/n equals p_d(J_d - I_d)"""
    mu_n = expectedSn(1000, p_d, n, d)
    J_d = np.asmatrix(np.ones((d,d)))
    I_d = np.asmatrix(np.identity(d)) 
    print(f'We should have {p_d*(J_d-I_d)}, and we have { mu_n / n}')
    
# test_verifyExpectedSn(0.5, 5, 4)

### Exercise 4

def SimulateV(n, Z, mu, p_d, d):
    """Simulates V = (S_n-mu)^T Z(S_n-mu)"""
    S_n = np.asmatrix(simulateSn(p_d, n, d))
    Z = np.asmatrix(Z)
    return np.asarray(np.transpose(S_n - mu)*Z*(S_n - mu))
    
def expectedV(trials, n, Z, mu, p_d, d):
    """Determines expected value of V = (S_n-mu)^T Z(S_n-mu) based on trials"""
    total = np.zeros((d,d))
    for _ in range(trials):
        total += SimulateV(n, Z, mu, p_d, d)
    return total / trials

def test_V():
    Z = np.identity(2)
    trials = 30
    p_d=1
    d =2 
    n =3
    assert (expectedV(trials, n, Z, expectedSn(trials, p_d, n, d), p_d, d) == np.zeros((2,2))).all(),"For p_d = 1, d =2 and n=3, V =[[0,0],[0,0]]"

# test_V()

### Exercise 5

def simulateL(Z, h, mu, ni):
    """Simulates lambda_max * (Z^{-1} + h*mu +vi)"""
    A = np.linalg.inv(Z) + h*mu + ni
    lambda_max = max(np.linalg.eig(A)[0])
    return lambda_max

# The function expectedL is extra, and was not required, however it is used to test the function simulateL

def expectedL(trials, n, Z, h, mu, ni, p_d, d):
    """Calculates expected value of lambda_max * (Z^{-1} + h*mu +vi) based on trials"""
    total = np.zeros((d,d))
    for _ in range(trials):
        total += simulateL(Z, h, mu, ni)
    return total/trials

def test_L():
    Z = np.identity(2)
    trials = 30
    p_d = 1
    d = 2 
    n = 3
    h = 1
    mu = expectedSn(trials, p_d, n, d)
    assert (expectedL(trials, n, Z, h, mu, expectedV(trials, n, Z, mu, p_d, d), p_d, d) == pytest.approx(np.zeros((2,2))+9*np.ones((2,2))-6*np.identity(2))),"For p_d = 1, d =2 and n=3, L =[[3,9],[9,3]]"

# test_L()

### Exercise 6

def generateZ(sigma, d):
    """Generate a positive semi definite matrix Z"""
    assert len(sigma) == d, "Sigma should be a d x d matrix"
    Z = np.zeros((d,d))
    for _ in range(d):
        G = np.asmatrix(rng.multivariate_normal(np.zeros(d), sigma))
        Z += np.transpose(G)*G   #Note that sampling from the multivariate_normal gives a row vector, therefore we switch the transpose multiplication
    return Z

def test_generateZ():
    sigma = np.identity(3)
    d = 3
    Z = generateZ(sigma,d)
    assert (np.linalg.eig(Z)[0] > 0).all(), 'not all eigenvalues are positive'

# test_generateZ()

### Exercise 7

# Options to implement:
# - generate random matrix, and multiply by its transpose, since a matrix A*A^T is always positive semi definite
#   Note: Not all possible matrices can be generated this way.

def generateZ_V2(d):
    """Generate a positive semi definite matrix Z"""
    Z = np.zeros((d,d))
    for row in range(d):
        for col in range(d):
            Z[row][col] = np.round(rng.uniform(0,1)*100) #Every entry is an integer between 0 and 100 (not all positive definite matrices can be generated)
    Z = np.asmatrix(Z)
    return Z*np.transpose(Z)

def test_generateZ_V2():
    Z = generateZ_V2(5)
    assert (np.linalg.eig(Z)[0] > 0).all(), 'not all eigenvalues are positive'

# test_generateZ_V2()

##################################
### DELIVERABLE
##################################

# Program 1:

def visualizeEigValErdosRenyi(d,p):
    A = ErdosRenyi(p,d)
    eigenvalues = np.linalg.eig(A)[0]
    plt.hist(eigenvalues)
    plt.show()
    return A, eigenvalues

# visualizeEigValErdosRenyi(3,1) 

# Program 2:    (NOT TESTED YET)

def estimateSFree(d, m, n, p, sigma):
    """Estimates an upperbound of ||S^n||_free based on the function generateZ()"""
    trials = 100 # used to determine the number of trials to compute mu and ni
    mu = expectedSn(trials,p,n,d) 
    minimum1 = 'no number assigned' # minimum when using h = 1
    minimum2 = 'no number assigned' # minimum when using h = -1
    # Computing minimum for h = 1
    for _ in range(m): 
        h = 1
        Z = generateZ(sigma, d)
        ni = expectedV(trials,n,Z,mu,p,d)
        L = simulateL(Z,h,mu,ni)
        if minimum1 == 'no number assigned' or L < minimum1:
            minimum1 = L
        h = -1
        L = simulateL(Z,h,mu,ni)
        if minimum2 == 'no number assigned' or L < minimum2:
            minimum2 = L
    return max(minimum1, minimum2)

# a = estimateSFree(10, 100, n=1, p=0.05, sigma=np.identity(10))

# Program 3:    (NOT TESTED YET)

# def estimateSFree_V2(d, m, n, p):
#     """Estimates an upperbound of ||S^n||_free based on the function generateZ()"""
#     trials = 100 # used to determine the number of trials to compute mu and ni
#     mu = expectedSn(trials,p,n,d) 
#     minimum1 = 'no number assigned' # minimum when using h = 1
#     minimum2 = 'no number assigned' # minimum when using h = -1
#     # Computing minimum for h = 1
#     for _ in range(m): 
#         h = 1
#         Z = generateZ_V2(d)
#         ni = expectedV(trials,n,Z,mu,p,d)
#         L = simulateL(Z,h,mu,ni)
#         if minimum1 == 'no number assigned' or L < minimum1:
#             minimum1 = L
#         h = -1
#         L = simulateL(Z,h,mu,ni)
#         if minimum2 == 'no number assigned' or L < minimum2:
#             minimum2 = L
#     return max(minimum1, minimum2)

# ### Exercise a)

# A = ErdosRenyi(0.3, 20)
# B = ErdosRenyi(0.3, 20)
# C = ErdosRenyi(0.3, 20)
# # drawGraph(C)

# ### Exercise b)

# #for d in [100, 1000]:
# #    for p in [0.2, 0.5, 0.8]:
# #        A = ErdosRenyi(p, d)
# #        eigenvalues = histOfEigenvalues(A, d, p)

# ### Exercise c) 
# # given values
# sigma = np.identity(10)
# d = 10
# trials =200
# p =0.7
# n=1

# # start experiment ex 6
# lambda1 = [] # h = 1
# lambda2 = [] # h = -1
# mu = expectedSn(trials,p,n,d) 
# test_length = 1000
# for m in range(test_length):
#     Z = generateZ(sigma, d)
#     ni = expectedV(trials,n,Z,mu,p,d)
#     h = 1
#     lambda1.append(simulateL(Z,h,mu,ni))
#     h = -1
#     lambda2.append(simulateL(Z,h,mu,ni))


# values = [max((min(lambda1[:i+1]), min(lambda2[:i+1]))) for i in range(test_length)]
# values

# plt.plot(range(1, test_length +1), values)
# plt.xlabel('$m$')
# plt.ylabel('Upperbound for $\|S_n\|_{free}$')
# plt.show()


# # start experiment ex 7
# lambda1 = [] # h = 1
# lambda2 = [] # h = -1
# mu = expectedSn(trials,p,n,d) 
# test_length = 1000
# for m in range(test_length):
#     Z = generateZ_V2(d)
#     ni = expectedV(trials,n,Z,mu,p,d)
#     h = 1
#     lambda1.append(simulateL(Z,h,mu,ni))
#     h = -1
#     lambda2.append(simulateL(Z,h,mu,ni))


# values = [max((min(lambda1[:i+1]), min(lambda2[:i+1]))) for i in range(test_length)]
# values

# plt.plot(range(1, test_length +1), values)
# plt.xlabel('$m$')
# plt.ylabel('Upperbound for $\|S_n\|_{free}$')
# plt.show()


# ### Exercise f)
d = 100

alphas = np.linspace(0.01, 3.3, 100)

estimates_1 = []
estimates_2 = []

# empirical confidence intervals 
est_1_lower = []
est_1_upper = []
est_2_lower = []
est_2_upper = []

counter = 0

for alpha in alphas:

    p_d = ((np.log(d))**alpha)/d

    if p_d < 0:
        p_d = 0
    elif p_d > 1:
        p_d = 1

    nRuns = 10

    adjacency_matrices_A = ErdosRenyi(p_d, d=100, nrErdos=nRuns)

    spectral_norms_list = []
    norms_list = []

    for x in adjacency_matrices_A:
        A = np.asmatrix(x)
        J_d = np.asmatrix(np.ones((d,d)))
        np.fill_diagonal(J_d, 0)
        I_d = np.asmatrix(np.identity(d))
        centered = A - p_d*(J_d-I_d)

        # for estimate 1
        centered_product = np.dot(np.transpose(centered), centered)
        eigenvalues = np.linalg.eig(centered_product)[0]
        spectral_norm_A = np.sqrt(max(eigenvalues))
        spectral_norms_list.append(spectral_norm_A)

        # for estimate 2
        frobenius_norm_centered = np.linalg.norm(centered, ord='fro')
        norms_list.append(frobenius_norm_centered)

    estimate_1_spectral_norm = np.mean(spectral_norms_list)
    estimate_2_free_norm = np.mean(norms_list)

    estimates_1.append(estimate_1_spectral_norm)
    estimates_2.append(estimate_2_free_norm)

    est_1_lower.append(np.percentile(spectral_norms_list, 2.5))
    est_1_upper.append(np.percentile(spectral_norms_list, 97.5))
    est_2_lower.append(np.percentile(norms_list, 2.5))
    est_2_upper.append(np.percentile(norms_list, 97.5))

    counter += 1
    print(counter/len(alphas))

fig, ax = plt.subplots()
ax.plot(alphas, estimates_1, "r-", label= "$\\hat{\\xi}_\\alpha$")
ax.plot(alphas, estimates_2, "b-", label= "$\\hat{\\zeta}_\\alpha$")
ax.fill_between(alphas, (est_1_lower), (est_1_upper), color ='r', alpha=.1)
ax.fill_between(alphas, (est_2_lower), (est_2_upper), color ='b', alpha=.1)
plt.xlabel("$\\alpha$")
plt.ylabel("estimate value")
plt.title("($\\alpha$, estimate) plot with confidence bands")
plt.legend()
plt.show()

# ### Exercise g)
def histOfEigenvalues_Gaussian(A, d):
    """Gives a histogram of the eigenvalues of the centered matrix of gaussian matrix A"""
    assert len(A) == d, "Length of matrix A must equal d" 
    A = np.asmatrix(A)
    centered = (A - np.mean(A)) / np.sqrt(np.var(A))
    eigenvalues = np.linalg.eig(centered)[0]
    nr_bins = int(round((max(eigenvalues)) - min(eigenvalues)))
    plt.hist(eigenvalues, bins=nr_bins, edgecolor = 'k')
    plt.title(f'Eigenvalues of centered random matrix (Gaussian entries) with d={d}')
    plt.ylabel('count')
    plt.xlabel('eigenvalues')
    plt.show()
    return eigenvalues

# size of to be generated matrix
n = 1000 
variance = 2 
correlation = 0.9 

# generate covariance matrix to control degree of dependency of the entries
cov_matrix = np.ones((n, n)) * correlation + (1-correlation) * np.identity(n)
cov_matrix = variance * cov_matrix

# generate large random matrix with gaussian dependent entries that have mean 0 and investigate eigenvalues
A = np.random.multivariate_normal(np.zeros(n), cov_matrix, n)
histOfEigenvalues_Gaussian(A, n)