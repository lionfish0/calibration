import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels
import pandas as pd
import numpy as np
from tensorflow_probability import distributions as tfd
from scipy.linalg import block_diag


def placeinducingpoints(X,S,C,M=16):
    """
    set up inducing point locations - evenly spaced.
    
    """
    Z = np.linspace(np.min(X[:,0]),np.max(X[:,0]),M)[None,:]
    Z = np.repeat(Z,S,0)
    Z = Z.flatten()[:,None]
    #duplicate for different sensors...
    Z = np.c_[Z,np.repeat(np.arange(S),M)]
    #add different components...
    newZ = np.zeros([0,3])
    for compi in range(C):
        newZ = np.r_[newZ,np.c_[Z,np.full(len(Z),compi)]]
    return newZ
    
    
class Kernel():
    def __init__(self,gpflowkernel):
        self.gpflowkernel = gpflowkernel
    def matrix(self,X1,X2):    
        cov = self.gpflowkernel.K(X1,X2) * ((X2[:,1][None,:]==X1[:,1][:,None]) & (X2[:,2][None,:]==X1[:,2][:,None]))
        return cov.numpy().astype(np.float32)

class BrokenMultiKernel(): #delete this class
    def __init__(self,gpflowkernels,indices):
        """
        gpflowkernels = a list of GPflow kernels.
          E.g. gpflowkernels = [gpflow.kernels.RBF(1.0,200),gpflow.kernels.Bias(1.0)]
        indices = a list of lists,
         each row refers to the one or more parameters.
         each element of each list refers to a sensor.
          E.g. indexes = [[0,0,0,0,1,1]]
        """
        self.gpflowkernels = gpflowkernels
        self.indices = indices
    def matrix(self,X1,X2):
        blockks = []
        for param,index in enumerate(self.indices):
            for idx,i in enumerate(index):
                blockks.append((self.gpflowkernels[i].K(X1[(X1[:,1]==idx) & (X1[:,2]==param)],X2[(X2[:,1]==idx) & (X2[:,2]==param)])).numpy())
        return block_diag(*blockks).astype(np.float32)
        
        
class TemporaryMultiKernel(): #delete this class
    def __init__(self,gpflowkernels,indices):
        self.gpflowkernel = gpflowkernels[0]
    def matrix(self,X1,X2):    
        cov = self.gpflowkernel.K(X1,X2) * ((X2[:,1][None,:]==X1[:,1][:,None]) & (X2[:,2][None,:]==X1[:,2][:,None]))
        return cov.numpy().astype(np.float32)
 

class MultiKernel():
    def __init__(self,gpflowkernels,indices):
        """
        A kernel in which each sensor and parameter is independent
        The covariance for each sensor and parameter is defined by the list of gpflow kernels.
        Specify one or more gpflow kernels in a list, and index these using the indices
        parameter.
        
        gpflowkernels = a list of GPflow kernels.
          E.g. gpflowkernels = [gpflow.kernels.RBF(1.0,10),gpflow.kernels.RBF(1.0,200),gpflow.kernels.Bias(1.0)]
        indices = a list of lists,
         each row refers to the one or more parameters.
         each element of each list refers to a sensor.
          E.g. indexes = [[0,0,0,0,1,1],[2,2,2,2,2,2]]
         in this example the first parameter for all the sensors are RBF kernels, for the first four sensors
         the lengthscale is just 10, but for the last two it is 200.
         The second parameter is modelled with a bias kernel for all the sensors.
        """
        self.gpflowkernels = gpflowkernels
        self.indices = indices
    def oldmatrix(self,X1,X2):    
        cov = np.zeros([len(X1),len(X2)])
        for index_param,indices_param in enumerate(self.indices): #loop over the parameters
            for index_sensor,index in enumerate(indices_param): #loop over the sensors
                #print(index_param,index_sensor,index)
                #print(self.gpflowkernels)
                cov+=self.gpflowkernels[index].K(X1,X2) * ((X2[:,1][None,:]==index_sensor) & (X1[:,1][:,None]==index_sensor) & (X2[:,2][None,:]==index_param) & (X1[:,2][:,None]==index_param))
                print(cov.shape)      
        return cov.numpy().astype(np.float32)
    def matrix(self,X1,X2):    
        cov = np.zeros([len(X1),len(X2)])
        for index_param,indices_param in enumerate(self.indices): #loop over the parameters
            for index_sensor,index in enumerate(indices_param): #loop over the sensors
                keep1 = np.where((X1[:,1]==index_sensor) & (X1[:,2]==index_param))[0]
                keep2 = np.where((X2[:,1]==index_sensor) & (X2[:,2]==index_param))[0]
                cov[np.ix_(keep1,keep2)] = self.gpflowkernels[index].K(X1[keep1,0:1],X2[keep2,0:1])
        return cov.astype(np.float32)
        

        
        
class SparseModel():
    def __init__(self,X,Z,C,k,jitter=1e-4):
        """
        A Gaussian process Sparse model for performing Variational Inference.
        
        Parameters:
        X = a (2*C*N) x 3 matrix
        each pair of rows, from the top and bottom half of the matrix refer to a pair
        of observations taken by two sensors at the same(ish) place and time.
        Within these two halves, there are C submatrices, of shape (Cx3), each one is
        for a particular component of the calibration function.
        Columns: time, sensor, component
        
        Z = a (2*C*M) x 3 matrix, same structure as for X.
        Columns: time, sensor, component        
        
        C = number of components/parameters the transform requires (e.g. a straight line requires two).
        k = the kernel object
        
        Note the constructor precomputes some matrices for the VI.
        
        This class doesn't hold the variational approximation's mean and covariance,
        but assumes a Gaussian that is used for sampling by calling 'get_samples'
        or 'get_samples_one_sensor'.
        These two methods take mu and scale, which describe the approximation.
        If you set scale to None then it assumes you are not modelling the covariance
        and returns a single 'sample' which is the posterior mean prediction.
        """
        
        self.jitter = jitter
        
        self.k = k
        self.Kzz = k.matrix(Z,Z)+np.eye(Z.shape[0],dtype=np.float32)*self.jitter
        self.Kxx = k.matrix(X,X)+np.eye(X.shape[0],dtype=np.float32)*self.jitter
        self.Kxz = k.matrix(X,Z)
        self.Kzx = tf.transpose(self.Kxz)
        self.KzzinvKzx = tf.linalg.solve(self.Kzz,self.Kzx)
        self.KxzKzzinv = tf.transpose(self.KzzinvKzx)
        self.KxzKzzinvKzx = self.Kxz @ self.KzzinvKzx
        self.C = C
        self.Npairs = int(X.shape[0]/(C*2)) #actual number of observation *pairs*
        self.N = int(X.shape[0]/C)
    def get_qf(self,mu,scale):   
        """
        TODO: We could, when scale is None, return self.Kxx - self.KxzKzzinvKzx.
        """ 
        qf_mu = self.KxzKzzinv @ mu
        if scale is None:
            qf_cov = None
        else:
            qf_cov = self.Kxx - self.KxzKzzinvKzx + self.KxzKzzinv @ getcov(scale) @ self.KzzinvKzx
        return qf_mu,qf_cov
    def get_samples(self,mu,scale,num=100):
        """
        Get samples of the function components for every observation pair in X.
        Returns a num x N x (C*2) matrix,
          where num = number of samples
                N = number of observation pairs
                C = number of components
        So for the tensor that is returned, the last dimension consists of the pairs of
        sensors, with each pair being one of the C components.
        
        If scale is set to None, then we return a single sample, of the posterior mean
        (i.e. we assume a dirac q(f).
        Returns 1 x N x (C*2) matrix.
        
        """                
        qf_mu,qf_cov = self.get_qf(mu,scale)
        if scale is None:
            return tf.transpose(tf.reshape(qf_mu,[2*self.C,self.Npairs]))[None,:,:]
            
        batched_mu = tf.transpose(tf.reshape(qf_mu,[2*self.C,self.Npairs]))
        batched_cov = []
        for ni in range(0,self.Npairs*self.C*2,self.Npairs):
            innerbcov = []
            for nj in range(0,self.Npairs*self.C*2,self.Npairs):
                innerbcov.append(tf.linalg.diag_part(qf_cov[ni:(ni+self.Npairs),nj:(nj+self.Npairs)]))
            batched_cov.append(innerbcov)
        samps = tfd.MultivariateNormalFullCovariance(batched_mu,tf.transpose(batched_cov)+tf.eye(2*self.C)*self.jitter).sample(num)
        return samps
    
    def get_samples_one_sensor(self,mu,scale,num=100):
        """
        Get samples of the function components for a sensor.
        Returns a num x N x (C) matrix,
          where num = number of samples
                N = number of observation pairs
                C = number of components
        So for the tensor that is returned, the last dimension consists of the pairs of
        sensors, with each pair being one of the C components.
        """
        
        
        qf_mu,qf_cov = self.get_qf(mu,scale)
        if scale is None:
            return tf.transpose(tf.reshape(qf_mu,[self.C,self.N]))[None,:,:]
            
        batched_mu = tf.transpose(tf.reshape(qf_mu,[self.C,self.N]))
        batched_cov = []
        for ni in range(0,self.N*self.C,self.N):
            innerbcov = []
            for nj in range(0,self.N*self.C,self.N):
                innerbcov.append(tf.linalg.diag_part(qf_cov[ni:(ni+self.N*2),nj:(nj+self.N)]))
            batched_cov.append(innerbcov)
        samps = tfd.MultivariateNormalFullCovariance(batched_mu,tf.transpose(batched_cov)+tf.eye(self.C)*self.jitter).sample(num)
        return samps
    
def getcov(scale):
    return tf.linalg.band_part(scale, -1, 0) @ tf.transpose(tf.linalg.band_part(scale, -1, 0))

class CalibrationSystem():

    def __init__(self,X,Y,Z,refsensor,C,transform_fn,gpflowkernels,kernelindices,likemodel='fixed',gpflowkernellike=None,likelihoodstd=1.0,jitter=1e-4,lr=0.02,likelr=None,minibatchsize=100,sideY=None):
        """
        A tool for running the calibration algorithm on a dataset, produces
        estimates of the calibration parameters over time for each sensor.
        
        The input refers to pairs of colocated observations. N = the number of
        these pairs of events.
        
        Parameters
        ----------
        X : An N x 3 or N x 4 matrix:
           If it is N x 3: The first column is the time of the observation
                           pair, the second and third columns are the indices
                           of the sensors. It is assumed the two measurements
                           are made at the same time.
           If it is N x 4: The first two columns are the times of the 
                           observations. The third and fourth columns are
                           the indices of the sensors.
        Y : An N x 2 matrix. The two columns are the observations from the two
        sensors at the colocation event.
        Z : Either an M x 1 matrix or an M*S x 2 matrix. The former just has
        the inducing input locations specfied in time. The latter also specifies
        their associated sensor in the second column.
        refsensor : a binary, S dimensional vector. 1=reference sensor.
                    (number of sensors inferred from this vector)
        C : number of components required for transform_fn. Scaling will require
        just one. While a 2nd order polynomial will require 3.
        transform_fn : A function of the form: def transform_fn(samps,Y),
           where samps's shape is: [batch (number of samples) 
                                     x number of observations (N) 
                                     x (number of components) (C)].
                 Y's shape is [number of observations (N) x 1].
        gpflowkernels = a list of GPflow kernels.
          E.g. gpflowkernels = [gpflow.kernels.RBF(1.0,200),gpflow.kernels.Bias(1.0)]
        kernelindices = a list (length C) of lists (number of sensors), 
         each row refers to the one or more parameters.
         each element of each list refers to a sensor.
          E.g. indexes = [[0,0,0,0,1,1]]
                
        likemodel : specifies how the likelihood is modelled.
        It can be one of four values:
          - fixed [default, uses the value in likelihoodstd]
          - [not yet] single [optimise a single value [TODO Not Implemented]]!!
          - distribution [uses gpflowkernellike]
          - process [uses gpflowkernellike]                
        likelihoodstd : The standard deviation of the likelihood function,
                which by default computes the difference between observation
                pairs (default=1.0). The self.likelihoodfn could be set to
                a different likelihood.
        jitter : Jitter added to ensure stability (default=1e-4).
        lr, likelr : learning rates.
        sideY : side information (humidity, temperature, etc. These are provided
        to the transform_fn)
                             
        TODO what happens if refsensor is integer?
                          if float64 used for others..
        
        """

        #internally we use a different X and Z:
        #we add additional rows to X and Z to account for the components we are
        #modelling. In principle the different components could have different
        #inducing points and kernels, but for simplicity we combine them.
        #These two matrices have three columns, the time, the sensor and the
        #component. They are constructed as below, with the sensor measurement
        #pairs in cosecutive submatrices, which is iterated over C times.
        #Time Sensor Component
        #  1    0    0
        #  2    0    0
        #  1    1    0
        #  2    1    0
        #  1    0    1
        #  2    0    1
        #  1    1    1
        #  2    1    1
        
        
        #Legacy support: Previously we assumed the pair of observations happened at the same time
        #so X only had one time column. This is still allowed, but a 2nd column is added here
        if X.shape[1]==3:
            X = np.c_[X[:,0],X[:,0],X[:,1],X[:,2]]
    
        self.likemodel = likemodel
        S = len(refsensor)
        self.C = C
        self.fullsideY = sideY
        self.fullY = Y
        self.k = MultiKernel(gpflowkernels,kernelindices)

        
        self.likelihoodstd = likelihoodstd
        self.minibatchsize = minibatchsize
        self.N = len(X)
        if (self.N<self.minibatchsize): self.minibatchsize=self.N        
        
        if Z.shape[1]==1:
            Ztemp = np.c_[np.tile(Z,[S,1]),np.repeat(np.arange(S),len(Z))]
        if Z.shape[1]==2:
            Ztemp = Z
        self.Z = np.c_[np.tile(Ztemp,[C,1]),np.repeat(np.arange(self.C),len(Ztemp))]
        
        if likemodel=='distribution' or likemodel=='process':
            assert gpflowkernellike is not None, "You need to specify the kernel to use a distribution or process"
            self.klike = Kernel(gpflowkernellike)
            
            self.Zlike = np.c_[Ztemp,np.repeat(0,len(Ztemp))]
            if likelr is None: likelr = lr * 4 #probably can optimise this a little quicker?
            self.likeoptimizer = tf.keras.optimizers.Adam(learning_rate=likelr,amsgrad=False)
        
        self.fullX = X   

        self.refsensor = refsensor.astype(np.float32)
        self.jitter = jitter
        self.transform_fn = transform_fn  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,amsgrad=False)    
        self.precompute()
        
    def precompute(self): 
        
        #definition of q(u)
        M = self.Z.shape[0]
        self.mu = tf.Variable(0.3*tf.random.normal([M,1]))
        self.scale = tf.Variable(np.tril(0.1*np.random.randn(M,M)+1.0*np.eye(M)),dtype=tf.float32)        
        #self.mu = tf.Variable(0*tf.random.normal([M,1]))
        #self.scale = tf.Variable(5*np.eye(M),dtype=tf.float32) #tf.Variable(0.2*np.tril(1.0*np.random.randn(M,M)+1.0*np.eye(M)),dtype=tf.float32)   
        
        if self.likemodel=='distribution' or self.likemodel=='process':
            Mlike = self.Zlike.shape[0]
            self.mulike = tf.Variable(0.0001*tf.random.normal([Mlike,1]))
            mu_u = tf.Variable(np.full([Mlike],-12),dtype=tf.float32)
            cov_u = tf.Variable(self.klike.matrix(self.Zlike,self.Zlike),dtype=tf.float32)
            self.pulike = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)
        else:
            self.mulike = None
        if self.likemodel=='process':
            self.scalelike = tf.Variable(1e-10*np.eye(Mlike),dtype=tf.float32)
        else:
            self.scalelike = None
        
        #parameters for p(u)
        mu_u = tf.zeros([M],dtype=tf.float32)
        cov_u = tf.Variable(self.k.matrix(self.Z,self.Z),dtype=tf.float32)
        self.pu = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)
        
        
    def computeforminibatch(self,justuserefs=False):    
        if justuserefs: #to start with we just train on data from pairings with reference sensors to help optimisation
            print("!!JUST USING REFERENCE ROWS!!")
            keep = np.where((np.isin(self.fullX[:,2],np.where(self.refsensor)[0])) | (np.isin(self.fullX[:,3],np.where(self.refsensor)[0])))[0]
            batch = np.random.choice(keep,self.minibatchsize,replace=False)
        else:
            if self.N == self.minibatchsize: #select all of them
                #print("minibatch is full dataset")
                batch = np.arange(self.N)
            else:
                batch = np.random.choice(self.N,self.minibatchsize,replace=False)

        self.X = np.c_[np.tile(np.r_[np.c_[self.fullX[batch,0],self.fullX[batch,2]],np.c_[self.fullX[batch,1],self.fullX[batch,3]]],[self.C,1]),np.repeat(np.arange(self.C),2*self.minibatchsize)] 
        self.Xlike = np.c_[np.r_[np.c_[self.fullX[batch,0],self.fullX[batch,2]],np.c_[self.fullX[batch,1],self.fullX[batch,3]]],np.repeat(0,2*self.minibatchsize)]  
        self.ref = tf.gather(self.refsensor,tf.transpose(tf.reshape(self.X[:(2*self.minibatchsize),1:2].astype(int),[2,self.minibatchsize])))
        self.Y = self.fullY[batch,:]
        if self.fullsideY is not None:
            self.sideY = self.fullsideY[batch,:]
        else:
            self.sideY = None
        
    def likelihoodfn_nonstationary(self,scaledA,scaledB,varparamA,varparamB):
        return tfd.Normal(0,0.00001+tf.sqrt(tf.exp(varparamA)+tf.exp(varparamB))).log_prob(scaledA-scaledB)    
    
    #def likelihoodfn(self,scaledA,scaledB):
    #    return tfd.Normal(0,self.likelihoodstd).log_prob(scaledA-scaledB)
        
    def likelihoodfn(self,scaledA,scaledB,ref):
        #assert False, "Not Implemented. Please inherit and implement"
        likelihoodstd = np.sqrt((self.likelihoodstd**2)*np.sum(1-ref,1))
        #likelihoodstd = 1e-4+((1-np.any(ref,1))*(self.likelihoodstd)).astype(np.float32)
        #likelihoodstd = np.sqrt((self.likelihoodstd**2)*np.sum(1-ref,1))
        #likelihoodstd = self.likelihoodstd/100+self.likelihoodstd*np.min(1-ref,1)
        #return tfd.Normal(0,0.001+likelihoodstd).log_prob((scaledA-scaledB)/(scaledA+scaledB))
        #return tfd.Normal(0,self.likelihoodstd).log_prob((scaledA-scaledB)/(scaledA+scaledB#))
        #return tfd.Normal(0,self.likelihoodstd).log_prob(scaledA-scaledB)
        #return tfd.Normal(0,likelihoodstd).log_prob(scaledA-scaledB)
        return tfd.Normal(0,likelihoodstd).log_prob((scaledA-scaledB)/(0.5*(scaledA+scaledB)))
        

    #@tf.function
    def run(self,its=None,samples=100,threshold=0.001,verbose=False):
        """ Run the VI optimisation.
        
        its: Number of iterations. Set its to None to automatically stop
        when the ELBO has reduced by less than threshold percent
        (between rolling averages of the last 50 calculations
        and the 50 before that).
        samples: Number of samples for the stochastic sampling of the
        gradient
        threshold: if its is None, this is the percentage change between
        the rolling average, over 50 iterations. Default: 0.001 (0.1%).
        """
        elbo_record = []
        it = 0
        if verbose: print("Starting Run")
        try:
            while (its is None) or (it<its):
                if verbose: print(".",end="")                
                it+=1
                with tf.GradientTape() as tape:
                    qu = tfd.MultivariateNormalTriL(self.mu[:,0],self.scale)
                    self.computeforminibatch(False) #it<its*0.25) #for first 25% just use reference sensors - this doesn't seem to help anything
                    if self.likemodel=='distribution' or self.likemodel=='process':
                        self.smlike = SparseModel(self.Xlike,self.Zlike,1,self.k)
                    self.sm = SparseModel(self.X,self.Z,self.C,self.k)
                    samps = self.sm.get_samples(self.mu,self.scale,samples)
                    scaled = tf.concat([self.transform_fn(samps[:,:,::2],self.Y[:,0:1],self.sideY),self.transform_fn(samps[:,:,1::2],self.Y[:,1:2],self.sideY)],2)
                    scaled = (scaled * (1-self.ref)) + (self.Y * self.ref)
                    if self.mulike is not None: #if we have non-stationary likelihood variance...
                        qulike = tfd.MultivariateNormalTriL(self.mulike[:,0],self.scalelike)              
                        like = self.smlike.get_samples(self.mulike,self.scalelike,samples)
                        ell = tf.reduce_mean(tf.reduce_sum(self.likelihoodfn_nonstationary(scaled[:,:,0],scaled[:,:,1],like[:,:,0]*(1-self.ref[:,0])-1000*self.ref[:,0],like[:,:,1]*(1-self.ref[:,1])-1000*self.ref[:,1]),1))
                    else: #stationary likelihood variance
                        ell = tfp.stats.percentile(tf.reduce_sum(self.likelihoodfn(scaled[:,:,0],scaled[:,:,1],self.ref),1), 50.0, interpolation='midpoint')
                    elbo_loss = -((len(self.fullX)*2)/len(self.X) * ell)+tfd.kl_divergence(qu,self.pu)
                    if self.likemodel=='process':
                        assert self.mulike is not None
                        assert self.scalelike is not None
                        elbo_loss += tfd.kl_divergence(qulike,self.pulike)
                    if self.likemodel=='distribution':
                        assert self.mulike is not None
                        elbo_loss -= self.pulike.log_prob(self.mulike[:,0])
                    if verbose: 
                        if it%20==0: print("%d (ELBO=%0.4f)" % (it, elbo_loss))
                    
                    if (self.mulike is None) or (it%50<25): #optimise latent fns
                        gradients = tape.gradient(elbo_loss, [self.mu,self.scale])
                        self.optimizer.apply_gradients(zip(gradients, [self.mu, self.scale]))  
                    else: #this optimises the likelihood...
                        if self.likemodel=='distribution':
                            gradients = tape.gradient(elbo_loss, [self.mulike])
                            self.likeoptimizer.apply_gradients(zip(gradients, [self.mulike]))
                        if self.likemodel=='process':
                            gradients = tape.gradient(elbo_loss, [self.mulike,self.scalelike])
                            self.likeoptimizer.apply_gradients(zip(gradients, [self.mulike,self.scalelike]))

                    elbo_record.append(elbo_loss)
                if its is None:
                    if it>100:
                        oldm = np.median(elbo_record[-100:-50])
                        m = np.median(elbo_record[-50:])
                        if np.abs((oldm-m)/((oldm+m)/2))<threshold:
                            #check that nothing weird's happened!
                            if np.std(elbo_record[-50:])<np.std(elbo_record[-100:-50]):
                                break
                                
        except KeyboardInterrupt:
            pass
        return np.array(elbo_record)
        
