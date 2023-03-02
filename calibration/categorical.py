import numpy as np
from calibration import CalibrationSystem, SparseModel
import gpflow
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class AltCalibrationSystem(CalibrationSystem):
    def __init__(self,X,Y,Z,refsensor,C,gpflowkernels,kernelindices,jitter=1e-6,lr=0.02,minibatchsize=100,sideY=None,priorp=None):
        Y = Y.astype(int)
        #Legacy support: Previously we assumed the pair of observations happened at the same time
        #so X only had one time column. This is still allowed, but a 2nd column is added here
        if X.shape[1]==3:
            X = np.c_[X[:,0],X[:,0],X[:,1],X[:,2]]
        
        super().__init__(X,Y,Z,refsensor,C,None,None,gpflowkernels,kernelindices,likemodel='fixed',
                         gpflowkernellike=None,jitter=jitter,lr=lr,likelr=None,
                         minibatchsize=minibatchsize,sideY=sideY)
        confsize = np.sqrt(C).astype(int)
        if priorp is None:
            priorp = np.ones(confsize)/confsize
        else:
            assert len(priorp)==confsize
        self.priorp = priorp
        
        assert np.max(Y)<confsize
        assert np.min(Y)>=0
        
    def likelihoodfn(self,samps,Y,ref):
        print("--------------------")
        confsize = np.sqrt(self.C).astype(int)
        exp_samps = tf.reshape(tf.exp(samps),[len(samps),len(Y),confsize,confsize,2])
        probs = exp_samps/tf.reduce_sum(exp_samps,2)[:,:,None,:,:]
        probs = tf.transpose(probs,perm=[0,2,3,1,4])
        probs = tf.where(ref==0,probs,tf.tile(tf.eye(confsize)[:,:,None,None],[1,1,len(Y),2]))
        probs = tf.transpose(probs,perm=[3,1,0,2,4])
        
        #for each sample & observation pair, probs is two square matrices (for each observer)
        #with p(obs1_species1|true_species_Y)
        #here we multiply them together to give
        #p(obs1_species1,obs2_species2|true_species_Y)
        
        probs = tf.gather(probs[:,:,:,:,0],Y[:,0],axis=1,batch_dims=1)*tf.gather(probs[:,:,:,:,1],Y[:,1],axis=1,batch_dims=1)
        
        #for each sample & observation pair, probs is a vector
        #each element is the probability of the two observations given the true species
        #print("probs_gathered",probs[0,0,:])
        #here we multiply p(obs1_species1,obs2_species2|true_species_Y) * p(true_species_Y)
        #to give p(obs1_species1,obs2_species2,true_species_Y)
        #then sum over the true_species
        #to give p(obs1_species1,obs2_species2)
        
        probs = tf.reduce_sum(probs*self.priorp,axis=2)
        
        return tf.math.log(probs) #scaledA-scaledB) #think this is more appropriate for the data
    #@tf.function
    def run(self,its=100,samples=100,threshold=0.001):
        elbo_record = []
        print("Starting Run")
        try:
            for it in range(its):
                print(".",end="")
                with tf.GradientTape() as tape:
                    qu = tfd.MultivariateNormalTriL(self.mu[:,0],self.scale)
                    #print(self.mu[:,0],self.scale)
                    self.computeforminibatch(False)
                    self.sm = SparseModel(self.X,self.Z,self.C,self.k)
                    samps = self.sm.get_samples(self.mu,self.scale,samples)
                    ls = self.likelihoodfn(samps,self.Y,self.ref)
                    ell = tf.reduce_mean(ls)*len(self.fullY)#tf.reduce_sum(ls,0))
                    elbo_loss = -ell+tfd.kl_divergence(qu,self.pu)
                    gradients = tape.gradient(elbo_loss, [self.mu,self.scale])
                    self.optimizer.apply_gradients(zip(gradients, [self.mu, self.scale]))  
                    elbo_record.append(elbo_loss)
                    print(elbo_loss)
        except KeyboardInterrupt:
            pass
        return np.array(elbo_record),samps
        
        
#helper fns that probably should be in the class

def compute_posterior_probs(cs,sensors=None,t=0,num_samps=100):
    """
    For the optimised calibration system, cs, compute the mean probabilities
    returning the confusion matrices...
    """
    if sensors is None:
        sensors = np.arange(len(cs.refsensor)) #all of them.
    C = cs.C
    allprobs = []
    #for si,refs in enumerate(cs.refsensor):
    #    if refs: continue
    for si in sensors:
        x = np.array([[t]])
        testX = np.zeros([0,3])
        for ci in range(C):
            tempX = np.c_[x,np.ones_like(x)*si,np.full_like(x,ci)]
            testX = np.r_[testX,tempX]#.astype(int)
        testsm = SparseModel(testX,cs.Z,C,cs.k)
        qf_mu,qf_cov = testsm.get_qf(cs.mu,cs.scale)
        samps = testsm.get_samples_one_sensor(cs.mu,cs.scale,num_samps)

        exp_samps = tf.reshape(tf.exp(samps),[num_samps,len(cs.priorp),len(cs.priorp)])
        #probs = exp_samps/tf.reduce_sum(exp_samps,2)[:,:,None]
        probs = exp_samps/tf.reduce_sum(exp_samps,1)[:,None,:]
        allprobs.append(probs)
    return allprobs
def print_confusion_matrices(allprobs,conf_matrices=None):
    np.set_printoptions(precision=0,suppress=True)
    if conf_matrices is None:
        conf_matrices = [None]*len(allprobs)
    for si,(probs,conf_mat) in enumerate(zip(allprobs,conf_matrices)):
        print("Person %d" % si)
        if conf_mat is not None:
            print("True, latent confusion matrix")
            print(conf_mat.T*100)
        print("--mean posterior--")
        print(np.mean(probs,0)*100)
        print("95% CIs:")
        print(np.sort(probs,axis=0)[2,:,:]*100,'\n\n',np.sort(probs,axis=0)[97,:,:]*100)
        print("-------------------")
        
        
       
