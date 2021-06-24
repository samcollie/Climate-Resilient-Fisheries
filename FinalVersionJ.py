# 3/2/19 Speedier VERSION
# 9/7/20 Adjust the population parameters and ranges of the states
# Additions to the previous version "BasicVersion":
# - Put Golden Section Search inside the for loop to allow use of jit over the main time loop
# - Used Gauss-Hermite Quadrature in place of the monte-carlo
#   which will allow better sampling from the normal distribution with fewer evals

#%% Import modules
import numpy as np
import pandas as pd
from numba import jit
from matplotlib import pyplot as plt
import time
#%%

###############################################################################
# Prelimnary Step... Create 2-D Interpolation Function.
# Explination: The value function is going to be a dimension of res X ares
# Need to be able to evaluate the continuation for intermediate values of b & a
# There are many ways to do this but I am going to use piecewise linear interpolation as its the easiest 
# to get your head around... The downside is that it will lead to non-linearities so we'll see what happens.
###############################################################################

@jit(nopython=True)
def findBounds(z,Z):
    # z is the scalar value, Z is the discretized array....
    # This requires that z is within the bounds of Z (i.e. z in [Z[0],Z[-1]])
    
    if (z >= Z[0]) and (z < Z[-1]):
        # Determine increment between elements of Z & divide
        increment = Z[1] - Z[0]
        lowerBound = int((z - Z[0])/ increment)
    
    # In case you are at the upper bound, use the last 2 elements
    elif z == Z[-1]:
        lowerBound = len(Z)-2
        
    return np.array([lowerBound,lowerBound+1])
    
@jit(nopython=True)
def weightedBounds(z,Z):
    # Given the bounds, decide how much weight to attach to upper and lower value
    # The wieghts are based on distance, so if z=5 falls between 0 & 10, weights will be 0.5 and 0.5
    bounds = findBounds(z,Z)
    values = Z[bounds]
    
    # Find the total distance between the upper & lower bound
    diff = values[1] - values[0]
    diff = np.array([diff,diff]) # You have to keep dimensions matching when using numba, otherwise this isn't neccesary
    
    weights = (diff - np.abs(values - z)) / diff    

    return (bounds,weights)

@jit(nopython=True)
def piecewiseLinear2d(x,y,X,Y,matrix):
    # (x,y) are the coordinates at which you want to evaluate (aka biomass & productivity, b&a)
    # X & Y are the associated discritezed vectors (aka B & A)
    # matrix is the 2d matrix you wish to interpolate...
    
    # Find the bounds & weights for x & y....
    (boundsX,weightsX) = weightedBounds(x,X)
    (boundsY,weightsY) = weightedBounds(y,Y)
    
    # Now simply take a weighted average of the 4 corners
    val = 0
    for i in range(2):
        for j in range(2):
            val += matrix[boundsX[i],boundsY[j]]*weightsX[i]*weightsY[j]
    return val

# Makes a matrix of zeros with dimensions xDim x yDim
@jit(nopython=True)
def zeros2d(xDim,yDim):
    z= np.array([[0. for j in range(yDim)] for i in range(xDim)])
    return z

#%%

###############################################################################
# STEP 0: DEFINE GLOBAL PARAMETERS
###############################################################################

# These items control precision/how long the program will run
###############################################################################
res = 100          # Resolution of the biomass state
convcrit = 1E-6   # Convergence criterion for solvers (smaller=more precise)
Tmax = 1000       # Maximum number of time iterations
ghRes = 3         # Number of Gauss-Hermite Nodes... it doesn't take many
###############################################################################

# Get Gauss Hermite Weights & Nodes from numpy
ghNodes, ghWeights = np.polynomial.hermite.hermgauss(ghRes)
ghWeights = ghWeights/np.sqrt(np.pi)


discount = 0.95            # Discount factor
sd_w = 0.2572118           # standard deviation of the process errors (variability in productivity (a)=0.2572118)
WS1 = 0.0002324            # weight at recruitment (tonnes)
gstar = 0.94269            # growth-survival term
Ra = 8.3939                # mean of the productivity parameter (a)
Rb = -1.69752e-04          # density-dependent coefficient of the stock-recruitment function
sd_obs = 0.3429491         # standard deviation of the Ricker stock-recruitment model 0.3429491

###############################################################################
# STEP 1: DEFINE STATES
###############################################################################

# Biomass State Space
bmin = 0
bmax = 4e4
B = np.linspace(bmin,bmax,res)

# A (dynamic parameter) state space...
ares = int(0.9*res)
amin = 6.6
amax = 9.5
A = np.linspace(amin,amax,num=ares)

# Guess the continuation value


##################################################################################
# STEP 2: DEFINE STATE TRANSITIONS
##################################################################################

@jit(nopython=True)
def aNext(a,w):
    
    aNext = a + w
    
    # Force a to stay within the bounds...
    if aNext > amax: 
        aNext = amax
    elif aNext < amin: 
        aNext = amin
        
    return aNext

@jit(nopython=True)
def bNext(a,b,h,v):
 
    stock = b*(1.-h)     
    bNext = stock*(gstar + WS1*np.exp(a + Rb*stock + v))

    # Check the bounds on bNext
    if bNext > bmax:
        bNext = bmax
    elif bNext < bmin:
        bNext = bmin
        
    return bNext

#%%
##################################################################################
# STEP 3: CALCULATE FUTURE REWARD
##################################################################################

# Uses Gauss-Hermite Quadrature to find the expectation of a normally distributed
# variable. Doing so in a grid to account for 2 sources of stochasticity (v & w)
    
# For triplit (a,b,h) & Continuation value V, transition to the next state &
# return future reward pi

@jit(nopython=True)
def futureReward(a,b,h,V):
    
    # Use Gauss-Hermite Quadrature to take the expection of normally distributed variables

    pi = 0.
    
    for wIdx,w in enumerate(ghNodes):
        
        w = w*sd_w
        wWeight = ghWeights[wIdx]
        aPrime = aNext(a,w)
        
        for vIdx,v in enumerate(ghNodes):
            
            v = v*sd_obs
            vWeight = ghWeights[vIdx]
        
            bPrime = bNext(a,b,h,v)
            
            # Interpolate Value Function & add to pi
            pi += piecewiseLinear2d(bPrime,aPrime,B,A,V)*wWeight*vWeight
        
    return  pi

#%%
#########################################################
# STEP 4: Update of utility over one time period
#########################################################

# Get the payoff of harvest h given state (a,b) & continuation value V
# a,b,h are scalar while V is res x ares
@jit(nopython=True)
def harvestRule(a,b,h,V):
    # Current reward is total harvest
    current_reward = b*h   
    future_reward = futureReward(a,b,h,V)
    total_reward = current_reward + discount*future_reward
    return total_reward


#%%
#########################################################
# STEP 5: Value Function Iteration
#########################################################
# Keeps going until the harvest rule converges (or Tmax is hit...)


@jit(nopython=True)
def valueIter():
    
    V = zeros2d(res,ares) # Initialize V at zero
    VNext = zeros2d(res,ares)
    
    # Define an empty harvest array to store harvest rule & Continuation Value
    H = zeros2d(res,ares)    
    
    # loop over time    
    for t in range(Tmax):

        # Loop over the state space (enumerate gives an index & the value of the iterable)
        for bIndex,b in enumerate(B):  
            
            for aIndex,a in enumerate(A):
                
                # Howard's Improvement algorithm
                # Only do the max step for 50 burn-ins then every 
                # 25th iteration
                if (t<50) or (t % 25 == 0):
                
                    # Golden Section Search for optimal harvest rule
                    # Take advantage of the fact that harvest is increasing in b
                    # and decreasing in a
                    if bIndex == 0:
                        w = 0. # Lower Boundary
                    else:
                        w = H[bIndex-1,aIndex]              
                       
                    if aIndex == 0:
                        z = 1.0 # Upper boundary (was 1.)
                    else:
                        z = H[bIndex,aIndex-1]
                    
    
                    diff = 1.  # arbitrary initial value greater than convcrit
                    while diff > convcrit:
    
                        # x & y are points between A & D defined by the golden ratio
                        x = w + ((3-(5**0.5))/2)*(z-w)
                        y = w + (((5**0.5)-1)/2)*(z-w)
    
                        # Evaluate the function at B & C
                        fx = harvestRule(a,b,x,V)
                        fy = harvestRule(a,b,y,V)
    
                        if fx >= fy:
                            z = y
                        else:
                            w = x
                        diff = np.abs(w-z)
    
                    # After convergence, store average of x&y as optimal harvest 
                    # (they should be pretty close at this point anyways)    
    
                    H[bIndex,aIndex] = (x+y)/2
                    VNext[bIndex,aIndex] = (fx+fy)/2
                        
                else:
                    VNext[bIndex,aIndex] = harvestRule(a,b,H[bIndex,aIndex],V)
        
        # Check For Convergence:
        if np.abs(VNext-V).max() < convcrit:
            print('Converged at iteration:')
            print(t)
            return (H,V)
        
        else:
            V = VNext
            VNext = zeros2d(res,ares)            
            
    print('Maximum Number of Iterations Reached')            
    return (H,V)

#%%
start = time.time()
H,V = valueIter()
end = time.time()

print('Processing Time: {}'.format(end-start))

#%%
##################################################################################
# STEP 6: PLOT THE RESULTS
##################################################################################

aaa = [0,int(ares/4),int(ares/2),int(ares*3/4),-1]

# Create a figure with 1 row & 1 column
fig, ax = plt.subplots(1,1)
ax.plot(B,H[:,aaa])
ax.set_title("Control rules for different values of a(t-1)")
ax.set_xlabel("Biomass")
ax.set_ylabel("Harvest Rate")
ax.legend(np.around(A[aaa],2), title='Productivity')

fig.savefig('Rainbow.pdf',bbox_inches='tight')

#%%
np.save("Hbase.npy", H)
pd.DataFrame(H).to_csv('Hbase.csv')
pd.DataFrame(V).to_csv('Vbase.csv')
