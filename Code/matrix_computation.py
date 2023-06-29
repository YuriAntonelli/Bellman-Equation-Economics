# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import math

# Parameters
A = 1.1
alpha = .75
w_0 = 1.5
w_1 = .19
w_2 = .03
w = .05
eta = 1
q = .05
beta = .95

# Grids
e = np.linspace(1, 500, 200) 
for i in range(len(e)):
    e[i] = round(e[i])
    
h = np.linspace(1, 70, 70)
for i in range(len(h)): 
    h[i] = round(h[i])


#-----------------------------------------------------#
#-STEP 1: Function to find the optimal "h*" given "e"-#
#-----------------------------------------------------#
def optimal_h(e_next):
    LHS = np.zeros(len(h))
    RHS = np.zeros(len(h))
    for i in range(len(h)):
        LHS[i] = A*h[i]**(alpha-1)*alpha*e_next**alpha 
        RHS[i] = w*e_next + w*e_next*w_1 + 2*w*e_next*w_2*h[i] - 80*w*e_next*w_2 
    difference = abs(RHS-LHS)
    h_opt = int(h[np.argmin(difference)])
    return h_opt

#----------------------------------------#
#-STEP 2: compute the objective function-#
#----------------------------------------#
def objValue(e_index, e_nextindex, e, h, V):
    e_today = e[e_index]
    e_nextday = e[e_nextindex]
    h_today = optimal_h(e_today)
    
    # different elements of the function 
    revenue = A*(e_today*h_today)**alpha
    costs = w*e_today*(w_0 + h_today + w_1*(h_today-40) + w_2*(h_today-40)**2)
    adj_costs = (eta/2)*(e_nextday-(1-q)*e_today)**2
    
    obj_function = revenue - costs - adj_costs + beta*V[e_nextindex]
    return obj_function

#-----------------------------#
#-STEP 3: compute the maximum-#
#-----------------------------#
def computeMax(e_index, e, h, V):
    solution = float('-inf')
    optindex = -1
    N = len(e)
    ub = N
    lb = 0
    while ub != lb:
        test = math.floor((ub+lb)/2)
        value = objValue(e_index, test, e, h, V)
        valuenext = float('-inf')
        if test+1 < N:
            valuenext = objValue(e_index, test+1, e, h, V)
        if value < valuenext:
            lb = test+1 
            if valuenext > solution:
                solution = valuenext
                optindex = test+1
        else:
            ub = test
            if value > solution:
                solution = value
                optindex = test
    return solution, optindex 

#----------------------------------------#
#-STEP 4: compute the new value function-#
#----------------------------------------#
def newValueFunction(e, h, V):
    N = len(e)
    TV = np.zeros(N)
    gamma = np.zeros(N)
    for i in range(N):
        TV[i], gamma[i] = computeMax(i, e, h, V)
    gamma = gamma.astype(int)
    return TV, gamma

#-----------------------------------------------------#
#-STEP 5: Horward improvement in closed form solution-#
#-----------------------------------------------------#
def howardImprovement(gamma, e, h, V): 
    N = len(gamma)
    F = np.zeros(N)
    for i in range(N):
        F[i] = objValue(i, gamma[i], e, h, V) - beta*V[gamma[i]]
        
    A = scipy.sparse.lil_matrix((N, N))
    for i in range(N):
        A[i, gamma[i]] = 1

    # solve the linear system 
    Vnext = scipy.sparse.linalg.spsolve(scipy.sparse.identity(N) - beta*A, F)
    return Vnext  

#----------------------------------#
#-STEP 6: value function iteration-# 
#----------------------------------#
def valueFunctionIteration(e, h):
    N = len(e)
    V = np.random.normal(0,1,N) 
    TV = np.zeros(N)
    Vnext = np.zeros(N)
    gamma = np.zeros(N) 
    dist = 10
    epsilon = .1
    loops = 0
    while dist > epsilon:
        TV, gamma = newValueFunction(e, h, V)
        Vnext = howardImprovement(gamma, e, h, V)
        dist = (abs(V-Vnext)).max()
        V = Vnext.copy()
        loops += 1 
        print(f'loop number {loops} with distance {dist}')
    gamma = gamma.astype(int)
    return V, gamma 

V_ast4, Gamma_ast4 = valueFunctionIteration(e, h)