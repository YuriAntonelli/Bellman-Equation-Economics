# -*- coding: utf-8 -*-
"""
All the files show the result by means of a single decision variable,
but I put the code (as comment) for the 2 decision variable solution at the 
end of each script
"""
import numpy as np
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
e = np.linspace(1, 500, 20) 
for i in range(len(e)): 
    e[i] = round(e[i])
    
e_dense= np.linspace(1, 500, 200) 
for i in range(len(e)): 
    e_dense[i] = round(e[i])
    
h = np.linspace(1, 70, 70) 
for i in range(len(h)):
    h[i] = round(h[i])


#--------------------------------#
#-STEP 1: Interpolation function-#
#--------------------------------#
def interpolation(e_next, e, V):
    if e_next <= e[0]:
        return V[0]
    elif e_next >= e[-1]:
        return V[-1]
    else:
        n = math.floor((e_next-e[0])/(e[1]-e[0])) 
        ebefore = e[n]
        eafter = e[n+1]
        vbefore = V[n]
        vafter = V[n+1]
        return vbefore + (e_next-ebefore)/(eafter-ebefore)*(vafter-vbefore)    

#-----------------------------------------------------#
#-STEP 2: Function to find the optimal "h*" given "e"-#
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
#-STEP 3: compute the objective function-#
#----------------------------------------#
def objValue(e_index, e_nextindex, e, h, e_dense, V):
    e_today = e[e_index]
    e_nextday = e_dense[e_nextindex]
    h_today = optimal_h(e_today)
    
    # different elements of the function 
    revenue = A*(e_today*h_today)**alpha
    costs = w*e_today*(w_0 + h_today + w_1*(h_today-40) + w_2*(h_today-40)**2)
    adj_costs = (eta/2)*(e_nextday-(1-q)*e_today)**2
    
    obj_function = revenue - costs - adj_costs + beta*interpolation(e_nextday, e, V)
    return obj_function

#-----------------------------#
#-STEP 4: compute the maximum-#
#-----------------------------#
def computeMax(e_index, e, h, e_dense, V):
    solution = float('-inf')
    optindex = -1
    N = len(e)
    ub = N
    lb = 0
    while ub != lb:
        test = math.floor((ub+lb)/2)
        value = objValue(e_index, test, e, h, e_dense, V)
        valuenext = float('-inf')
        if test+1 < N:
            valuenext = objValue(e_index, test+1, e, h, e_dense, V)
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
#-STEP 5: compute the new value function-#
#----------------------------------------#
def newValueFunction(e, h, e_dense, V):
    N = len(e)
    TV = np.zeros(N)
    gamma = np.zeros(N)
    for i in range(N):
        TV[i], gamma[i] = computeMax(i, e, h, e_dense, V)
    return TV, gamma

#----------------------------------#
#-STEP 6: value function iteration-#
#----------------------------------#
def valueFunctionIteration(e, h, e_dense):
    N = len(e)
    V = np.random.normal(0,1,N) 
    TV = np.zeros(N)
    gamma = np.zeros(N) 
    dist = 10
    epsilon = .1
    loops = 0
    while dist > epsilon:
        TV, gamma = newValueFunction(e, h, e_dense, V)
        dist = max(abs(V-TV))
        V = TV.copy()
        loops += 1 
        print(f'loop number {loops} with distance {dist}')
    gamma = gamma.astype(int)
    return V, gamma 

V_ast2, Gamma_ast2 = valueFunctionIteration(e, h, e_dense)

#--------------------------------------------------------------------------

##### SOLUTION WITH 2 DECISION VARIABLES

"""
e = np.linspace(1, 500, 20) 
for i in range(len(e)): 
    e[i] = round(e[i])

edense = np.linspace(1, 500, 100) 
for i in range(len(e)):
    e[i] = round(e[i])

h = np.linspace(1, 70, 70) 
for i in range(len(h)): 
    h[i] = round(h[i])
    
#---------------------------------------#
#-Step 1: linear interpolation function-#
#---------------------------------------#
# The idea is to make a function to linearly interpolate only with respect 
# to the variable "e", so keeping a fixed hindex. 
# That's because our V function is bivariate but for the purpose of
# the assignment we don't need a bilinear interpolation
def lin_interpolation(enext, hnext, e, h, V):
    n_h = math.floor((hnext-h[0])/(h[1]-h[0])) # index of hnext
    
    if enext <= e[0]:
        return V[0, n_h]
    elif enext >= e[-1]:
        return V[-1, n_h]
    else:
        n_e = math.floor((enext-e[0])/(e[1]-e[0]))
        ebefore = e[n_e]
        eafter = e[n_e+1]
        vbefore = V[n_e, n_h]
        vafter = V[n_e+1, n_h]  
        return vbefore + (enext-ebefore)/(eafter-ebefore)*(vafter-vbefore)   

#----------------------------------------#
#-Step 2: compute the objective function-#
#----------------------------------------#
def objValue(e_index, h_index, e_nextindex, h_nextindex, e, h, edense, V):
    e_today = e[e_index]
    e_nextday = edense[e_nextindex]
    h_today = h[h_index]
    h_nextday = h[h_nextindex]
    
    revenue = A*(e_today*h_today)**alpha
    costs = w*e_today*(w_0 + h_today + w_1*(h_today-40) + w_2*(h_today-40)**2)
    adj_costs = (eta/2)*(e_nextday-(1-q)*e_today)**2
    
    obj_function = revenue - costs - adj_costs + beta*lin_interpolation(e_nextday, h_nextday, e, h, V)
    return obj_function

#-----------------------------#
#-Step 3: compute the maximum-#
#-----------------------------#
def computeMax(e_index, h_index, e, h, edense, V):
    solution = float('-inf')
    optindex = [-1, -1]
    N_h = len(h)
    N_e = len(edense)
    
    for i in range(N_h):
        ub = N_e
        lb = 0
        while ub != lb:
            test = math.floor((ub+lb)/2)
            value = objValue(e_index, h_index, test, i, e, h, edense, V)
            valuenext = float('-inf')
            if test+1 < N_e:
                valuenext = objValue(e_index, h_index, test+1, i, e, h, edense, V)
            if value < valuenext:
                lb = test+1 
                if valuenext > solution:
                    solution = valuenext
                    optindex = [test+1, i]
            else:
                ub = test
                if value > solution:
                    solution = value
                    optindex = [test, i]

    return solution, optindex 

#----------------------------------------#
#-Step 4: compute the new value function-#
#----------------------------------------#
def newValueFunction(e, h, edense, V):
    N_e = len(e)
    N_h = len(h)
    TV = np.zeros(shape=(N_e, N_h))
    gamma = []
    for i in range(N_e):
        for j in range(N_h):
            result = computeMax(i, j, e, h, edense, V)
            TV[i,j] = result[0]
            gamma.append(result[1])
    return TV, gamma 

#----------------------------------#
#-Step 5: value function iteration-#
#----------------------------------#
def valueFunctionIteration(e, h, edense):
    N_e = len(e)
    N_h = len(h)
    V = np.zeros(shape=(len(e), len(h))) 
    for r in range(len(e)):
        V[r, ] = np.random.normal(0,1, len(h))
    TV = np.zeros(shape=(N_e, N_h))
    gamma = [] 
    dist = 10
    epsilon = 1
    loops = 0
    while dist > epsilon:
        TV, gamma = newValueFunction(e, h, edense, V)
        dist = (abs(V-TV)).max()
        V = TV.copy()
        loops += 1 
        print(f'loop number {loops} with distance {dist}')
    for i in range(len(gamma)):
        gamma[i] = [int(x) for x in gamma[i]]
    return V, gamma
    
V_ast2, Gamma_ast2 = valueFunctionIteration(e, h, edense)
"""





