import numpy as np

def getrealpolution(t,loc):
    return (loc[0]*30+10*np.cos(1+loc[1]*4))*np.sin(t/2.3)+np.cos(1+t/10)*30+loc[0]*5+loc[1]*3+50

def getstaticsensortranform(t,pol,Nrefs,sensor,noisescale):##25
    if sensor<Nrefs: return pol
    #noisescale = 1*(1+np.cos(t*0.002*(1+sensor/3))+np.sin(t*0.001*(1+sensor/10)))
    v = pol*(1+0.5*np.sin(1+sensor)+0.2*np.cos(1+t*0.003*(1+np.cos(sensor))+sensor)) + np.random.randn()*noisescale
    #v[v<1,:]=1
    if v<1:
        v=1
    return v

def getmobilesensortranform(t,pol,sensor,noisescale):##25
    #noisescale = 1*(1+np.cos(t*0.002*(1+sensor/3))+np.sin(t*0.001*(1+sensor/10)))
    v = pol*(1.2+0.2*np.sin(4+sensor)+0.4*np.cos(2+t*0.01*(1+0.5*np.cos(4+sensor))+3+sensor)) + np.random.randn()*noisescale
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
