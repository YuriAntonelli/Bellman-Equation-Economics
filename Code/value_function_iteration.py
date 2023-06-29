# -*- coding: utf-8 -*-
"""
All the files show the result by means of a single decision variable,
but I put the code (as comment) for the 2 decision variable solution at the 
end of each script
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
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

#----------------------------------------------------#
#-STEP1: Function to find the optimal "h*" given "e"-#
#----------------------------------------------------#
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

#----------------------------#
#-STEP 4: New value function-#
#----------------------------#
def newValueFunction(e, h, V):
    N = len(e)
    TV = np.zeros(N)
    gamma = np.zeros(N)
    for i in range(N):
        TV[i], gamma[i] = computeMax(i, e, h, V)
    return TV, gamma

#----------------------------------#
#-STEP 5: Value function iteration-#
#----------------------------------#
def valueFunctionIteration(e, h):
    N = len(e)
    V = np.random.normal(0,1,N) 
    TV = np.zeros(N)
    gamma = np.zeros(N) 
    dist = 10
    epsilon = .1
    loops = 0
    while dist > epsilon:
        TV, gamma = newValueFunction(e, h, V)
        dist = max(abs(V-TV))
        V = TV.copy()
        loops += 1 
        print(f'loop number {loops} with distance {dist}')
    gamma = gamma.astype(int)
    return V, gamma 

V_ast1, Gamma_ast1 = valueFunctionIteration(e, h)

#------------------------------------------------------------------------------

#-------#
#-Plots-#
#-------#

# 1) Value Function
plt.figure(figsize=(12,8))
plt.title('Value function', fontsize=15)
plt.plot(e, V_ast1)
"""
The value function appears to be concave and 
monotonically increasing with respect to the
initial state "e"
"""

# 2) e_0 and e_1
plt.figure(figsize=(12,8))
plt.title('state-action', fontsize=15)
plt.plot(e, Gamma_ast1)

# 3) add the 45 degree line
plt.figure(figsize=(12,8))
plt.title('state-action', fontsize=15)
plt.plot(e, Gamma_ast1)
plt.plot(e, e)
plt.xlim([0, 25])
plt.ylim([0, 25])
"""
It can been seen that, according to this model parameters 
and these discretization grids, the "e" in the long run will
converge towards e* which is around 13. Of course, in order to have 
more precise results, which can change also a lot, it would be required 
to simulate the model with much denser grids
"""

#----------------------------------------------------------------------

##### SOLUTION WITH 2 DECISION VARIABLES

"""
# Grids
e = np.linspace(1, 500, 50) # number of workers
for i in range(len(e)): # let's make them integers
    e[i] = round(e[i])
    
h = np.linspace(1, 70, 50) # hours worked
for i in range(len(h)): # let's make them integers
    h[i] = round(h[i])

#----------------------------------------#
#-Step 1: compute the objective function-#
#----------------------------------------#
def objValue(e_index, h_index, e_nextindex, h_nextindex, e, h, V):
    e_today = e[e_index]
    e_nextday = e[e_nextindex]
    h_today = h[h_index]
    
    # different elements of the function 
    revenue = A*(e_today*h_today)**alpha
    costs = w*e_today*(w_0 + h_today + w_1*(h_today-40) + w_2*(h_today-40)**2)
    adj_costs = (eta/2)*(e_nextday-(1-q)*e_today)**2
    
    obj_function = revenue - costs - adj_costs + beta*V[e_nextindex, h_nextindex]
    return obj_function

#-----------------------------#
#-Step 2: compute the maximum-#
#-----------------------------#
def computeMax(e_index, h_index, e, h, V):
    solution = float('-inf')
    optindex = np.array([-1, -1])
    N_h = len(h)
    N_e = len(e)
    
    for i in range(N_h):
        ub = N_e
        lb = 0
        while ub != lb:
            test = math.floor((ub+lb)/2)
            value = objValue(e_index, h_index, test, i, e, h, V)
            valuenext = float('-inf')
            if test+1 < N_e:
                valuenext = objValue(e_index, h_index, test+1, i, e, h, V)
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
#-Step 3: compute the new value function-#
#----------------------------------------#
def newValueFunction(e, h, V):
    N_e = len(e)
    N_h = len(h)
    TV = np.zeros(shape=(N_e, N_h))
    gamma = []
    for i in range(N_e):
        for j in range(N_h):
            result = computeMax(i, j, e, h, V)
            TV[i,j] = result[0]
            gamma.append(result[1])
    return TV, gamma 

#----------------------------------#
#-Step 4: value function iteration-#
#----------------------------------#
def valueFunctionIteration(e, h):
    N_e = len(e)
    N_h = len(h)
    
    V = np.zeros(shape=(len(e), len(h))) # value function
    for r in range(len(e)):
        V[r, ] = np.random.normal(0,1, len(h))
    
    TV = np.zeros(shape=(N_e, N_h))
    gamma = [] 
    dist = 10
    epsilon = 1
    loops = 0
    while dist > epsilon:
        TV, gamma = newValueFunction(e, h, V)
        dist = (abs(V-TV)).max()
        V = TV.copy()
        loops += 1 
        print(f'loop number {loops} with distance {dist}')
    # makes gamma int list
    for i in range(len(gamma)):
        gamma[i] = [int(x) for x in gamma[i]]
    
    return V, gamma   

V_ast1, Gamma_ast1 = valueFunctionIteration(e, h)

#---------------------------------------------------------------------

#-------#
#-Plots-#
#-------#

# 1) VALUE FUNCTION 
e = e.astype(int)
h = h.astype(int)
# Value function and e0, k0
# Create a meshgrid from the dimensions of the 2D array
x, y = np.meshgrid(e, h)
# Create a 3D figure and axis
fig = plt.figure(figsize = (15, 10))
plt.title('Value Function', fontsize=20)
ax = fig.add_subplot(111, projection='3d')
# Plot the 2D array as a surface
ax.plot_surface(x, y, V_ast1)
# Set the axis labels
ax.set_xlabel('e (number of employers)')
ax.set_ylabel('h (hours worked)')
ax.set_zlabel('Value Function')
"""





















