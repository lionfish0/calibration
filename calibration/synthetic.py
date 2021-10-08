import numpy as np
import pandas as pd

def getrealpolution(t,loc):
    return (loc[0]*30+10*np.cos(1+loc[1]*4))*np.sin(t/2.3)+np.cos(1+t/10)*30+loc[0]*5+loc[1]*3+50

def getstaticsensortranform(t,pol,Nrefs,sensor,noisescale):##25
    if sensor<Nrefs: return pol
    #noisescale = 1*(1+np.cos(t*0.002*(1+sensor/3))+np.sin(t*0.001*(1+sensor/10)))
    #v = pol*(1+0.5*np.sin(1+sensor)+0.2*np.cos(1+t*0.003*(1+np.cos(sensor))+sensor)) + np.random.randn()*noisescale
    v = pol*(1+0.5*np.sin(1+sensor)+0.2*np.cos(1+t*0.002*(1+0.5*np.cos(sensor))+sensor)) + np.random.randn()*noisescale
    #v[v<1,:]=1
    if v<1:
        v=1
    return v

def getmobilesensortranform(t,pol,sensor,noisescale):##25
    #noisescale = 1*(1+np.cos(t*0.002*(1+sensor/3))+np.sin(t*0.001*(1+sensor/10)))    
    #v = pol*(1.2+0.2*np.sin(4+sensor)+0.4*np.cos(2+t*0.01*(1+0.5*np.cos(4+sensor))+3+sensor)) + np.random.randn()*noisescale
    v = pol*(1.2+0.2*np.sin(4+sensor)+0.4*np.cos(2+t*0.005*(1+0.5*np.cos(4+sensor))+3+sensor)) + np.random.randn()*noisescale    
    if v<1:
        v=1
    #v[v<1,:]=1
    return v
    #pol*(1+0.2*np.sin(4+sensor)+0.4*np.cos(2+t*0.01*(1+0.5*np.cos(4+sensor))+3+sensor)) + np.random.randn()*noisescale
    #return pol + np.random.randn()*noisescale
    
def generate_synthetic_dataset(Nstatic, 
                               Nmobile, 
                               Ttotal, 
                               Nrefs, 
                               Nvisitsperdayref,
                               Nvisitsperday,
                               staticsensornoise,
                               mobilesensornoise,
                               Nsamps):
    """
    Nvisitsperdayref = expected number of times each mobile sensor visits a static reference sensor.
    Nvisitsperday = expected number of times each mobile sensor visits a static sensor.
    Ttotal = number of time steps (hours)
    Nsamps = number of samples per colocation event

    in reality the boda drivers have their own 'patches' which mean that they are more likely
    to visit sensors in their own patch.
    We build our synthetic data by temporarily building a simulation of the locations
    of the static sensors and the centres of the mobile (boda-boda) motorbike taxi
    activities. The boda-bodas are organised to have particular waiting areas, known
    as stages, around the city. Typically a boda-boda will have one stage they are
    allowed to wait at. They are also therefore more likely to visit sensors in their
    part of the city.

    We simulate this by selecting randomly locations for these stages and then assign
    a probability of visiting proportional to  the inverse distance.

    With only two or three reference sensors arranged across Kampala we pay the boda-boda
    drivers to visit them once a week to recalibrate the mobile sensors. Future papers
    will explore the optimum sequence of visits.
    
    Ignoring night/day, we simply note that each hour has a probability of approximately
    Nvisitsperday/24. We multiply this by the probabilities assigned to each static sensor
    (for being visited by each mobile sensor).
    """                            
    refsensor = np.zeros(Nstatic+Nmobile)
    refsensor[0:Nrefs]=1
    
    statics = np.random.rand(Nstatic,2)
    mobilecentres = np.random.rand(Nmobile,2)

    invdistances = 1/np.sqrt(np.sum((mobilecentres-statics[:,None,:])**2,2))
    rawprobs = invdistances/np.sum(invdistances,0)    
    
    probs = (Nvisitsperday/24)*rawprobs*(1-refsensor[:Nstatic])[:,None] + (Nvisitsperdayref/24)*rawprobs*(refsensor[:Nstatic])[:,None]
    
    X = np.zeros([0,3])
    Y = np.zeros([0,2])
    trueY = np.zeros([0])
    for t in range(Ttotal):
        stats,mobs = np.where(np.random.random_sample(probs.shape)<probs)
        tempt = t

        for stat,mob in zip(stats,mobs):
            pol = getrealpolution(t,statics[stat])
            polstat = [getstaticsensortranform(t,pol,Nrefs,stat,noisescale=staticsensornoise) for i in range(Nsamps)]
            polmobile = [getmobilesensortranform(t,pol,mob,noisescale=mobilesensornoise) for i in range(Nsamps)]

            newX = np.c_[np.linspace(tempt,tempt+0.33,Nsamps)[:,None],np.full(Nsamps,stat),np.full(Nsamps,mob+Nstatic)]
            newY = np.c_[polstat,polmobile]
            X = np.r_[X,newX]
            Y = np.r_[Y,newY]
            trueY = np.r_[trueY,pol]
            tempt+=0.33
            
    Y[Y<1]=1
    trueY[trueY<1]=1
    return refsensor,X,Y,trueY,statics,mobilecentres
    
###Bee example of categorical data###

def genguess(conf_matrix,truebee):
    return np.random.choice(conf_matrix.shape[0],1,p=conf_matrix[truebee,:])
    
def syntheticA():
    """
    e.g. call: 
      priorp,conf_matrices,truebees = syntheticA()    
    """
    #priorp = [0.7,0.145,0.145,0.01] # [0.25,0.25,0.25,0.25]
    priorp = [0.50 , 0.17, 0.15, 0.1, 0.08]
    #priorp = [0.8, 0.2]
    priorp = np.array(priorp)/np.sum(priorp)
    truebees = np.random.choice(len(priorp),1000,p=priorp)
    #we initially assume that the confusion matrices are stationary.
    conf_matrices = []
    #conf_matrices.append(np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]))
    c = np.eye(len(priorp))
    c[0,:]=1
    conf_matrices.append(c)
    conf_matrices.append(np.ones([len(priorp),len(priorp)])+np.eye(len(priorp))) #complete random+a bit of accuracy.
    conf_matrices.append(np.random.rand(len(priorp),len(priorp))+np.eye(len(priorp))*5)
    conf_matrices.append(np.random.rand(len(priorp),len(priorp))+np.eye(len(priorp))*0.5)
    conf_matrices.append(np.eye(len(priorp)))
    for i in range(len(conf_matrices)): #normalising
        conf_matrices[i] = (conf_matrices[i].T/np.sum(conf_matrices[i],1)).T
    return priorp, conf_matrices, truebees
def syntheticB():
    """
    e.g. call: 
      priorp,conf_matrices,truebees = syntheticB()    
    """
    priorp = [0.95,0.05]
    truebees = np.random.choice(2,1000,p=priorp)
    conf_matrices = []
    conf_matrices.append(np.ones([2,2]))
    conf_matrices.append(np.eye(2))
    for i in range(len(conf_matrices)): #normalising
        conf_matrices[i] = (conf_matrices[i].T/np.sum(conf_matrices[i],1)).T
    return priorp, conf_matrices, truebees

def syntheticC():
    """
    e.g. call: 
      priorp,conf_matrices,truebees = syntheticC()    
    """
    priorp = [0.90,0.05,0.05]
    truebees = np.random.choice(3,1000,p=priorp)
    conf_matrices = []
    conf_matrices.append(np.ones([3,3]))
    conf_matrices.append(np.eye(3))
    for i in range(len(conf_matrices)): #normalising
        conf_matrices[i] = (conf_matrices[i].T/np.sum(conf_matrices[i],1)).T
    return priorp, conf_matrices, truebees

def syntheticD():
    """
    e.g. call: 
      priorp,conf_matrices,truebees = syntheticD()    
    """
    priorp = [0.6,0.2,0.2]
    truebees = np.random.choice(3,1000,p=priorp)
    conf_matrices = []
    conf_matrices.append(np.array([[1,1,1],[0,1,0],[0,0,1]]))
    conf_matrices.append(np.eye(3))
    for i in range(len(conf_matrices)): #normalising
        conf_matrices[i] = (conf_matrices[i].T/np.sum(conf_matrices[i],1)).T
    return priorp, conf_matrices, truebees

def buildXY_from_D(D):
    X = []
    Y = []
    for i,d in enumerate(D):
            for ci in range(len(d)):
                for cj in range(ci+1,len(d)):
                    if (~np.isnan(d[ci])) and (~np.isnan(d[cj])):
                        X.append([i,ci,cj])
                        Y.append([d[ci],d[cj]])  
    X = np.array(X)
    Y = np.array(Y)                        
    return X,Y
    
def build_D_from_csv(csvfilename,removerare):
    df = pd.read_csv(csvfilename,index_col=0)
    df = df.replace(removerare,[np.NaN]*len(removerare))
    names = np.array(list(set(df.to_numpy().flatten())))
    names = names[[n!='nan' for n in names]]
    freqs = np.array([np.sum(df.to_numpy().flatten()==i) for i in names]).astype(float)
    priorp = freqs/np.sum(freqs)
    namereplacementvalues = np.arange(len(names))
    df = df.replace(names,namereplacementvalues)
    df = df.replace(-1,np.NaN)
    truebees = df['ground truth'].to_numpy().copy()
    return df.to_numpy(), priorp, truebees
    
def gen_synthetic_observations(priorp,conf_matrices,truebees):
    X = []
    Y = []
    D = np.full([len(truebees),len(conf_matrices)],np.NAN)
    for i,tb in enumerate(truebees):
        p = 0.6
        for j,m in enumerate(conf_matrices):
            if np.random.rand()<p:
                D[i,j] = genguess(m,tb)
                p = 0.5
                #if j==1: p = 0 #we stop any colocations between #1 and #2[ref]
            else:
                p*=2**(1/(len(conf_matrices)-1))
    X,Y = buildXY_from_D(D)
   
    refsensor = np.zeros(len(conf_matrices))
    refsensor[-1]=1
    return X,Y,refsensor,D



