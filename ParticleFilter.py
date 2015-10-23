"""
Generic Sequential Importance Resampling Particle Filter for tracking

Allows matrices and parameters to potentially change at every time step,
instead of assuming certain variables are fixed (e.g. your observation matrix,
H, could change in time if certain sensor modalities go on/offline; or e.g. 
sensor statuses could change, changing the errors associated with their 
measurements, thus changing the measurement covariance matrix).

The propagation step requires some assumed process noise distribution. 
Currently, the process noise is hardcoded as zero mean Gaussian noise,
which is NOT a required assumption. Could either just hardcode other noise
distribution, or feed in a string parameter, e.g. 'Gaussian' or 'Rayleigh' and
then a dictionary of degrees of freedom for that distribution.
Same idea for measurement noise step.

The DynamicAdjust method is a customizable method that could be 
developed in order to dynamically adjust various parameters, like
the number of particles, or the threshold for doing resampling.


Python 2.7.6 |Anaconda 1.9.1 (64-bit)| (default, Nov 11 2013, 10:49:15) [MSC v.1500 64 bit (AMD64)] on win32
NumPy 1.9.3, SciPy 0.16.0, Matplotlib 1.4.3
"""






import numpy as np



class ParticleFilter():
    
    
    def __init__(self, mu_t0, StateCovariance_t0, nParticles_t0):
        
        #Initialize particles as multidimensional Gaussian cloud #Shape: D x nParticles
        self.mu = mu_t0
        self.StateCovariance = StateCovariance_t0
        self.nParticles = nParticles_t0
        self.P = np.random.multivariate_normal(self.mu, self.StateCovariance, size=(self.nParticles)).T
        #(no reason that it has to be Gaussian: e.g. if have absolutely no prior knowledge,
        #could do multivariate uniform distribution to uniformly sample the state space)
        #self.P = multivariate_uniform(..., size=(self.nParticles)).T 
        
        #Initialize weights uniformly:
        self.weights = np.repeat(1./self.nParticles,self.nParticles)
        
        #Initialize variable placeholders:
        self.nIteration = 0
        self.ResampledIterationsList = [] #List of iterations that resampled. Indexing starting at 0
        self.SIR_threshold = None
        self.nParticles_next_t = None
        self.F = None
        self.dt = None
        self.measurement_vector = None
        self.H_particles = None
        self.MeasurementCovariance = None
        self.MeasurementCovarianceINV = None
        self.weight_multipliers = None
        self.ProcessCovariance = None
        self.difference = None
        
        
        
    def Propagate(self, dt, F_func_list_t, Process_Covariance_t):
        #Check dimensions correct:
        message1 = None; message2 = None
        if len(F_func_list_t) != self.mu.size:
            message1 = 'F_func_list_t must be list of functions with D elements\n'+\
            'where D = number of dimensions of state space.\n' +\
            'Also, each of the D functions must be explicitly functions of ALL the D variables (even if not actually depend on some variables)'
        if Process_Covariance_t.shape != self.ProcessCovariance.shape: #!!!!!this line assumes state space dimensions constant in time
            message2 =  'Process_Covariance_t must be a square symmetric array of shape D x D\n'+\
            'where D = number of dimensions of state space.'
        if message1 != message2 != None:
            raise Exception(message1 + '\n'*2 + message2)
        self.ProcessCovariance = Process_Covariance_t
        self.dt = dt
        #Propagate the array of particles in state space using the list of propagation transformation functions at the current timestep:
        F_t  = np.zeros(self.P.shape) #Shape: D x nParticles
        for particle in xrange(self.nParticles):
            F_t[:,particle] = np.array([f(np.append(self.P[:,particle],dt)) for f in F_func_list_t])
        self.F = F_t
        #Below assumes Gaussian PROCESS noise, which is not a required assumption. Can change this for other noise profiles.
        self.P = self.F + np.random.multivariate_normal(np.zeros(self.F.shape[0]),self.ProcessCovariance,size=(self.nParticles)).T #!!!!!this line assumes state space dimensions constant in time

        
        
    def Reweight(self, measurement_vector_t, H_func_list_t, MeasurementCovariance_t):
        #Check dimensions correct:
        M = len(H_func_list_t) 
        #check M = meas vec size
        if MeasurementCovariance_t.shape != (M,M):
            message = 'H_func_list_t must be list of functions with M elements\n'+\
            'where M = number of dimensions of observation space.\n'+\
            'Also, each of the M functions must be explicitly functions of ALL the D state space variables (even if not actually depend on some variables).\n'+\
            'Usually, M < D, since usually measuring a subset of the state space dimensions.\n\n'+\
            'MeasurementCovariance_t must be a square symmetric array of shape M x M'
            raise Exception(message)
        self.MeasurementCovariance = MeasurementCovariance_t
        self.measurement_vector = measurement_vector_t
        #get Moore-Penrose pseudo-inverse of covariance matrix:
        self.MeasurementCovarianceINV = np.linalg.pinv(self.MeasurementCovariance)
        #Build H based on list of functions:
        #H_func_list_t is a LIST of functions. Each of the functions is a function of ALL the variables, even if given variable not explicitly used in function expression
        
        #Make the array H_particles_t of particles in OBSERVATION space using the list of observation transformation functions at the current timestep:
        measurements_repeated = np.repeat(self.measurement_vector.reshape(self.measurement_vector.size,1),self.nParticles,axis=1)
        H_particles_t  = np.zeros(measurements_repeated.shape)
        for particle in xrange(self.nParticles):
            H_particles_t[:,particle] = np.array([h(self.P[:,particle]) for h in H_func_list_t])
        self.H_particles = H_particles_t 
        #Array of differences (distances) between each particle and the measurement
        #Below assumes Gaussian MEASUREMENT noise, which is not a required assumption. Can change this for other noise profiles.
        self.difference = np.abs(self.H_particles - measurements_repeated)
        mahalanobis2 = np.dot(np.dot(self.difference.T,self.MeasurementCovarianceINV),self.difference)#is an nParticles x nParticles matrix
        self.weight_multipliers = np.exp(-.5*np.diag(mahalanobis2)) #Don't need gaussian amplitude prefactor since next step is normalization
        #Update and normalize the weights:
        self.weights *= self.weight_multipliers 
        self.weights /= np.sum(self.weights)
       
        

    def SIR(self):
        """
        Sequential Importance Resampling (SIR):
        If some metric of the particle weights is not above some threshold
        (in this case the metric is [1 / sum of particle weights]^2), this
        indicates that the current ensemble of particles is not tracking the 
        state very well, so the particles are reinitialized randomly according
        to a probability distribution that is a function of the particle weights.
        """
        
        def DynamicAdjust(self):
            """
            Define custom method to dynamically adjust parameters.
            E.g.: increase number of particles when uncertainty is elevated,
            or dynamically adjust threshold as a function of nParticles, when
            nParticles is variable. Or keep a running histoy of recent weight
            multiplier values and consider some function of them in adjusting 
            threshold...
            
            **Right now, implemented to keep nParticles constant, and keep a 
            constant SIR_threshold of ~.6*nParticles
            """

            #Modify this function as desired, or leave as is to use constants:

            #Just keep nParticles constant:
            self.nParticles_next_t = self.nParticles

            #Calculate the SIR_threshold:
            frac = 2./(1. + 5.**.5) #Arbitrary value between (0,1) : this is the reciprocal Golden Ratio
            #Some people use .6, some people use 2/3.
            self.SIR_threshold = round(frac*self.nParticles) #Use arbitrary K * nParticles threshold
            
            #Could also use the variables: self.nIteration, self.ResampledIterationsList
            #and look at trends in resampling, e.g. if you resampled last 3 in a row iterations,
            #your particles are possibly not tracking the actual state well, so increase nParticles
            #to have better chance of improved tracking, or lower the threshold to make resampling less likely
            #to give the resampled particles a few iterations to try and better fit the state vector.
            #or possibly increase the variance of your particle distribution to sample a larger range of values

        #Do the DynamicAdjust algorithm to determine threshold and nParticles_next_t:
        DynamicAdjust(self)
        
        #Check if need to resample:
	   #Inverse square metric is used in:
	   #Tutorial on Particle Filters by Sanjeev Arulampalam et. al, 2002:
	   #http://www.wisdom.weizmann.ac.il/~vision/courses/SML_2003/tutorialParticleFilters.pdf
        if 1./np.sum(self.weights**2) < self.SIR_threshold:
            #Take nParticles_next_t draws with replacement from current particles, with probabilities given by their weights:
            inds = np.random.choice(np.arange(self.nParticles), size=self.nParticles_next_t, replace=True, p=self.weights)
            #Make new matrix of state space particles, with not necessarily same nParticles as previously:
            self.P = self.P[:,inds]
            #Update the nParticles variable:
            self.nParticles = self.nParticles_next_t  
            #Uniformly reinitialize the weights:
            self.weights = np.repeat(1./self.nParticles,self.nParticles)
            #Append this iteration number to the list of which iterations used a resampling step:
            self.ResampledIterationsList += [self.nIteration]
        
        
        
    def Estimate(self):
        """
        Calculate the estimated state vector and state covariances
        """

        #Different estimates:
        #Could do mode, median, mean, or other.
        #Bayesian MMSE estimator is just weighted arithmetic mean of particle states:
        
        #Get the WEIGHTED array of particle states to use in both calculations:
        weights_repeat = np.repeat(np.expand_dims(self.weights,axis=1),self.P.shape[0],axis=1)
        weighted_P = self.P*weights_repeat
        #(on the first time step, and on time steps where SIR was just done,
        #the weights are all 1/nParticles, so the mean = weighted mean,
        #and same for the covariance matrix calculation)

        #The estimated state vector is the weighted mean of all particles:
        self.mu = np.mean(weighted_P,axis=1)

        #The estimated state covariance matrix is the weighted covariance matrix of all particles:
        self.StateCovariance = np.cov(weighted_P) #np.cov default assumption is rows are variables, columns are observations
        
        
        
    def Update(self, dt, F_func_list_t, Process_Covariance_t, measurement_vector_t, H_func_list_t, MeasurementCovariance_t):
        #Arguably, you can do these steps in different orders.
        #Using this ordering means that the estimated state vector and state covariance
        #are measured AFTER (possibly) doing the SIR step.
    
        #The Propagate + Reweight steps comprise the Measurement update step. The
        #particles are propagated in state space according to the elapsed time,
        #dt, and the transition model, F_func_list_t. The particle states are
        #converted to measurement space according to H_func_list_t and 
        #compared to the actual noisy measurement_vector_t, and reweighted
        #based on their Mahalonobis distances.
    
        #You could logically return the state vector estimate here, but it might
        #be a bad estimate if many of the weights are small. Instead, return the
        #state estimate (and covariance estimate) after the SIR step. If the
        #SIR threshold condition is not met, then the weights are fine and the
        #ordering of Estimate vs. SIR is irrelevant. However, if many weights are small,
        #the SIR step is done, and the returned estimate is based on the resampled particles.
    
        self.Propagate(dt, F_func_list_t, Process_Covariance_t)
        self.Reweight(measurement_vector_t, H_func_list_t, MeasurementCovariance_t)
        self.SIR()
        self.Estimate()
        self.nIteration += 1
