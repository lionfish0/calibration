from surprise.dataset import Dataset
from surprise import Reader
import pandas as pd
import numpy as np

def collaborative_filtering_SVD(D,maxval,refsensor):
    results = []
    for species_id in range(maxval):
        bees = []
        people = []
        guesses = []
        for person in range(D.shape[1]):
            for bee in range(D.shape[0]):
                if np.isnan(D[bee,person]): continue
                bees.append(bee)
                people.append(person)
                guesses.append(D[bee,person])
        guesses = np.array(guesses)
        guesses = (guesses==species_id).astype(float)

        guesses_dict = {'bee': bees,
                        'person': people,
                        'guess': guesses}
        guesses_df = pd.DataFrame(guesses_dict)

        reader = Reader(rating_scale=(0, 1))
        dataset = Dataset.load_from_df(guesses_df[['person', 'bee', 'guess']], reader)
        from surprise import SVD
        svd = SVD()
        trainset = dataset.build_full_trainset()
        svd.fit(trainset)
        #print([svd.predict(np.where(refsensor)[0][0],testbee).est for testbee in range(D.shape[0])])
        results.append([svd.predict(np.where(refsensor)[0][0],testbee).est for testbee in range(D.shape[0])])
    colfil_results = np.array(results).T
    colfil_results = (colfil_results.T/np.sum(colfil_results,1).T).T
    return colfil_results
    
    
from scipy import optimize

def get_trust_weights(D,maxval):
    """
    We could just find the most popular class, weighted by some 'trust' weight.
    This fn assumes the last column of D is the reference/true label.
    """
    def getnegscore(trust):
        score = 0
        for ds in D:
            if np.isnan(ds[-1]): continue #this doesn't have a true label...
            guess = np.zeros(maxval)
            for personi,d in enumerate(ds[:-1]):
                if np.isnan(d): continue
                guess[int(d)]+=trust[personi]  #*priorp[int(d)]
            if (np.argmax(guess)==ds[-1]): score += 1
        return -score


    best = None
    for its in range(20):
        x0 = np.random.rand(D.shape[1]-1)
        res = optimize.minimize(getnegscore,x0,method='Nelder-Mead')
        if best is None: best = res
        if res['fun']<best['fun']: best=res
    trust = best['x']
    return trust

def printresults(results):
    print("=====PERCENT CORRECT=====")
    print("Calibration Method            %0.1f%% correct" % (100*np.mean(results['correct'])))
    print("Collaborative Filtering       %0.1f%% correct" % (100*np.mean(results['colfil_guess'])))
    print("Baseline (most guessed)       %0.1f%% correct" % (100*np.mean(results['baseline_mostguessed'])))
    print("Baseline (\", trust weighted)  %0.1f%% correct" % (100*np.mean(results['trust_guess'])))
    print("Baseline (\", prior weighted)  %0.1f%% correct" % (100*np.mean(results['baseline_mostguessedtimespriorp'])))
    print("Baseline (most common)        %0.1f%% correct" % (100*np.mean(results['baseline_mostcommon'])))
    print("Baseline (most common B)      %0.1f%% correct" % (100*np.max([np.mean(blmc) for i,blmc in results['baseline_mostcommon_debug'].items()])))

    print("======NLPD======")
    print("Calibration Method            %0.2f" % -np.sum(np.log(results['correctprob'])))
    print("Collaborative Filtering       %0.2f" % -np.sum(np.log(results['colfilprob_guess'])))
    print("Baseline (most guessed)       %0.2f" % -np.sum(np.log(results['baselineprob_mostguessed'])))
    print("Baseline (\", trust weighted)  %0.2f" % -np.sum(np.log(results['trustprob_guess'])))
    print("Baseline (\", prior weighted)  %0.2f" % -np.sum(np.log(results['baselineprob_mostguessedtimespriorp'])))
    print("Baseline (most common)        %0.2f" % -np.sum(np.log(results['baselineprob_mostcommon'])))

    
def evaluate(D,truevals,priorp,allprobs,colfil_results,refsensor,trust):
    results = {}
    results['correct'] = []
    results['baseline_mostcommon'] = []
    results['baseline_mostguessedtimespriorp'] = []
    results['baseline_mostguessed'] = []
    results['colfil_guess'] = []
    results['trust_guess'] = []

    results['correctprob'] = []
    results['baselineprob_mostcommon'] = []
    results['baselineprob_mostguessedtimespriorp'] = []
    results['baselineprob_mostguessed'] = []
    results['colfilprob_guess'] = []
    results['trustprob_guess'] = []
    results['baseline_mostcommon_debug'] = {}

    for vali,(ds,t) in enumerate(zip(D,truevals)):
        if np.isnan(t):
            continue #if we don't even know the true value of a row, just skip.
        pOs = np.ones([allprobs[0].shape[1],len(priorp)])
        if ~np.isnan(ds[refsensor==1][0]): 
            continue

        t = t.astype(int)
        for i,d in enumerate(ds):
            if np.isnan(d): continue
            pOs*=allprobs[i][vali,:,int(d),:] #p(observation=value|species) - not a distribution!

        #pOs = p(observation_1=value|species) * p(observation_2=value|species) ...
        #    = p(obs1=v,obs2=v|species)
        #  p(obs1=v,obs2=v|species) p(species) / sum_s(p(obs1=v,obs2=v|species)p(species))
        # =p(obs1=v,obs2=v,species) / p(obs1=v,obs2=v)
        # =p(species|obs1=v,obs2=v)
        #the mean is just because we have lots of samples.
        pofs = np.mean(np.array(pOs*priorp).T/np.sum(pOs*priorp,1),1) #prob of species|observations

        results['correct'].append(t==np.argmax(pofs))
        results['correctprob'].append(pofs[t])

        #0.2 added as otherwise these are infinitely bad.
        temp = 0.2+np.array([np.sum(ds==i) for i in range(len(priorp))])*priorp
        temp = temp/np.sum(temp)
        results['baseline_mostguessedtimespriorp'].append(t==np.argmax(temp))
        results['baselineprob_mostguessedtimespriorp'].append(temp[t])

        tempB = 0.2+np.array([np.sum(ds==i) for i in range(len(priorp))])
        tempB=tempB/np.sum(tempB)
        results['baseline_mostguessed'].append(t==np.argmax(tempB))
        results['baselineprob_mostguessed'].append(tempB[t])

        for i in range(len(priorp)):
            if i not in results['baseline_mostcommon_debug']: results['baseline_mostcommon_debug'][i] = []
            results['baseline_mostcommon_debug'][i].append(t==i)
        results['baseline_mostcommon'].append(t==np.argmax(priorp))
        results['baselineprob_mostcommon'].append(priorp[t])

        results['colfil_guess'].append(np.argmax(colfil_results[vali,:])==t)
        results['colfilprob_guess'].append(colfil_results[vali,t])

        guess = np.zeros(len(priorp))
        for personi,d in enumerate(ds[:-1]):
            if np.isnan(d): continue
            guess[int(d)]+=trust[personi]
        results['trust_guess'].append(np.argmax(guess)==t)

        guess = guess + 0.2
        normalised_guesses = (guess/np.sum(guess))
        results['trustprob_guess'].append(normalised_guesses[t])

        if temp[t]==0: print(temp)
        #print(ds,pofs,t,np.argmax(pofs),np.argmax(temp),np.argmax(tempB),temp[t],tempB[t])
    return results
