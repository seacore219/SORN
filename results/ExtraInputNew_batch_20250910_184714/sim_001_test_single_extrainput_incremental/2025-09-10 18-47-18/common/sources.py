from __future__ import division
from pylab import *
import random
import itertools
import utils
utils.backup(__file__)
import synapses
import numpy as np 


class AbstractSource(object):
    def __init__(self):
        """
        Initialize all relevant variables.
        """
        raise NotImplementedError
    def next(self):
        """
        Returns the next input
        """
        raise NotImplementedError
    def global_range(self):
        """
        Returns the maximal global index of all inputs
        """
        raise NotImplementedError
    def global_index(self):
        """
        TODO check if this is really the case!
        Returns the current global (unique) index of the current input
        "character"
        """
        raise NotImplementedError
    def generate_connection_e(self,N_e):
        """
        Generates connection matrix W_eu from input to the excitatory
        population

        Parameters:
            N_e: int
                Number of excitatory units
        """
        raise NotImplementedError
    def generate_connection_i(self,N_e):
        """
        Generates connection matrix W_iu from input to the inhibitory
        population

        Parameters:
            N_e: int
                Number of excitatory units
        """
        raise NotImplementedError


class CountingSource(AbstractSource):
    """
    Source for the counting task.
    Different of words are presented with individual probabilities.
    """
    def __init__(self, words, probs, N_u_e, N_u_i, avoid=False):
        self.word_index = 0  
        self.ind = 0         
        self.words = words   
        self.probs = probs   
        self.N_u_e = int(N_u_e)  
        self.N_u_i = int(N_u_i)
        self.avoid = avoid
        
        # Get unique characters from all words
        all_chars = "".join(words)
        # Use set to get unique characters, then sort them
        self.alphabet = sorted(list(set(all_chars)))
        self.N_a = len(self.alphabet)
        # Create lookup dictionary mapping each character to an index
        self.lookup = dict(zip(self.alphabet, range(self.N_a)))
        
        # Debug print to verify
        print("Alphabet:", self.alphabet)
        print("Lookup:", self.lookup)
        
        self.glob_ind = [0]
        self.glob_ind.extend(cumsum(map(len, words)))
        self.predict = self.predictability()
        self.reset()

    # THIS METHOD MUST BE INDENTED AS PART OF CountingSource CLASS
    def generate_connection_e(self, N_e):
        W = zeros((N_e, self.N_a))

        available = set(range(N_e))
        for a in range(self.N_a):
            temp = random.sample(available, self.N_u_e)
            W[temp, a] = 1
            if self.avoid:
                available = available.difference(temp)
        if '_' in self.lookup:
            W[:, self.lookup['_']] = 0

        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        ans = synapses.create_matrix((N_e, self.N_a), c)
        ans.W = W
        return ans

    # THIS METHOD MUST ALSO BE INDENTED AS PART OF CountingSource CLASS
    def generate_connection_i(self, N_i):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        ans = synapses.create_matrix((N_i, self.N_a), c)
        W = zeros((N_i, self.N_a))
        if N_i > 0:
            available = set(range(N_i))
            for a in range(self.N_a):
                temp = random.sample(available, self.N_u_i)
                W[temp, a] = 1
            if '_' in self.lookup:
                W[:, self.lookup['_']] = 0
        ans.W = W
        return ans

    # ... rest of the methods (char, index, next_word, etc.) ...

    def char(self):
        word = self.words[self.word_index]
        return word[self.ind]

    def index(self):
        character = self.char()
        if character not in self.lookup:
            print("Character '%s' not in lookup!" % character)
            print("Available characters:", self.lookup.keys())
            print("Current word index:", self.word_index)
            print("Current character index:", self.ind)
        ind = self.lookup[character]
        return ind

    def next_word(self):
        self.ind = 0
        w = self.word_index
        p = self.probs[w, :]
        # Fix the deprecation warning by using np.where instead of find
        self.word_index = np.where(rand() <= cumsum(p))[0][0]

    def next(self):
        self.ind = self.ind + 1
        string = self.words[self.word_index]
        if self.ind >= len(string):
            self.next_word()
        ans = zeros(self.N_a)
        ans[self.index()] = 1
        return ans

    def reset(self):
        # Initialize to a valid state
        self.word_index = 0  # Start with first word
        self.ind = -1        # Will be incremented to 0 on first next() call

    def global_index(self):
        return self.glob_ind[self.word_index] + self.ind

    def global_range(self):
        return self.glob_ind[-1]

    def trial_finished(self):
        return self.ind + 1 >= len(self.words[self.word_index])

    def predictability(self):
        """
        TODO What's happening here? Success of random guessing?
        """
        temp = self.probs
        for n in range(10):
            temp = temp.dot(temp)
        final = temp[0, :]
        # Let's assume that all words have unique initial letters
        probs = map(len, self.words)
        probs = array(probs)
        probs = (probs + self.probs.max(1) - 1) / probs
        return sum(final * probs)

class TrialSource(AbstractSource):
    """
    This source takes any other source and gives it a trial-like
    structure with blank periods inbetween stimulation periods
    The source has to implement a trial_finished method that is True
    if it is at the end of one trial
    """
    def __init__(self,source,blank_min_length,blank_var_length,
                 defaultstim,resetter=None):
        assert(hasattr(source,'trial_finished')) # TODO for cluster
        self.source = source
        self.blank_min_length = blank_min_length
        self.blank_var_length = blank_var_length
        self.reset_blank_length()
        self.defaultstim = defaultstim
        self.resetter = resetter
        self._reset_source()
        self.blank_step = 0

    def reset_blank_length(self):
        if self.blank_var_length > 0:
            self.blank_length = self.blank_min_length\
                                + randint(self.blank_var_length)
        else:
            self.blank_length = self.blank_min_length

    def next(self):
        if not self.source.trial_finished():
            return self.source.next()
        else:
            if self.blank_step >= self.blank_length:
                self.blank_step = 0
                self._reset_source()
                self.reset_blank_length()
                return self.source.next()
            else:
                self.blank_step += 1
                return self.defaultstim


    def _reset_source(self):
        if self.resetter is not None:
            getattr(self.source,self.resetter)()

    def global_range(self):
        return source.global_range() # TODO +1 ?

    def global_index(self):
        if self.blank_step > 0:
            return -1
        return self.source.global_index()

    def generate_connection_e(self, N_e):
        return self.source.generate_connection_e(N_e)

    def generate_connection_i(self, N_i):
        return self.source.generate_connection_i(N_i)

class AndreeaCountingSource(AbstractSource):
    """
    This was only for debugging purposes - it resembles her matlab code
    perfectly
    """
    def __init__(self,sequence,sequence_U,pop,train):
        self.pop = pop-1
        self.seq = sequence[0]-1
        self.seq_u = sequence_U[0]-1
        self.t = -1
        # change m,n,x to make them identical


        if train:
            self.seq[self.seq==2] = 80
            self.seq[self.seq==3] = 2
            self.seq[self.seq==4] = 3
            self.seq[self.seq==80] = 4
            self.seq_u[self.seq_u>13] = (self.seq_u[self.seq_u>13]%7)+7
            self.lookup = {'A':0,'B':1,'M':2,'N':3,'X':4}
        else:
            self.seq[self.seq==2] = 7
            self.seq[self.seq==9] = 2
            self.seq[self.seq==3] = 5
            self.seq[self.seq==10] = 3
            self.seq[self.seq==4] = 6
            self.seq[self.seq==11] = 4
            self.lookup = {'A':0,'B':1,'X':7,'M':5,'N':6,'C':2,'D':3,'E':4}

            #~ self.lookup = {'A':0,'B':1,'X':2,'M':3,'N':4,'C':9,'D':10,'E':11}

            self.alphabet = 'ABCDEMNX'
            self.words = ['AXXXXXM','BXXXXXN','CXXXXXN','CXXXXXM','DXXXXXN','DXXXXXM','EXXXXXN','EXXXXXM']
            self.glob_ind = [0]
            self.glob_ind.extend(cumsum(map(len,self.words)))
        self.N_a = self.seq.max()+1

    def next(self):
        self.t += 1
        tmp = zeros((self.N_a))
        tmp[self.seq[self.t]] = 1
        return tmp
    def global_range(self):
        return self.seq_u.max()
    def global_index(self):
        return self.seq_u[self.t]
    def generate_connection(self,N_e):
        W = np.zeros((N_e,self.N_a))
        for i in range(self.N_a):
            if i <= 4: #--> A,B,X,M,N
                W[self.pop[:,i],i] = 1
            if i == 9:
                W[self.pop[0:2,0],i] = 1
                W[self.pop[2:10,1],i] = 1
            if i == 10:
                W[self.pop[0:5,0],i] = 1
                W[self.pop[5:10,1],i] = 1
            if i == 11:
                W[self.pop[0:8,0],i] = 1
                W[self.pop[8:10,1],i]  = 1
        self.W = W
        return W

class NoSource(AbstractSource):
    """
    No input for the spontaneous conditions

    Parameters:
        N_i: int
            Number of input units
    """
    def __init__(self,N_i=1):
        self.N_i = N_i
    def next(self):
        return np.zeros((self.N_i))

    def global_range(self):
        return 1

    def global_index(self):
        return -1

    def generate_connection_e(self,N_e):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        tmpsyn = synapses.create_matrix((N_e,self.N_i),c)
        tmpsyn.set_synapses(tmpsyn.get_synapses()*0)
        return tmpsyn

    def generate_connection_i(self,N_i):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        return synapses.create_matrix((N_i,self.N_i),c)

class RandomSource(AbstractSource):
    """
    Poisson input spike trains.
    """
    def __init__(self, firing_rate, N_neurons, connection_density,eta_stdp):
        """
        Initialize the source

        Parameters:
            firing_rate: double
                The firing rate of all input neurons
            N_neurons: int
                The number of poisson input units
            connection_density: double
                Density of connections from input to excitatory pop
            eta_stdp: double
                STDP rate for the W_eu matrix
        """
        self.rate = firing_rate
        self.N = N_neurons
        self.density = connection_density
        self.eta_stdp = eta_stdp

    def next(self):
        print("[RandomSource] Using correct next() method")
        return rand(self.N)<=self.rate
    def global_range(self):
        return 1
    def global_index(self):
        return 0
    def generate_connection_e(self,N_e):
        print("[DEBUG] generate_connection_e called")
        print("[DEBUG] N_e =", N_e)
        print("[DEBUG] self.N =", self.N)
        print("[DEBUG] self.density =", self.density)
        print("[RandomSource] Using correct generate_connection_e")
        c = utils.Bunch(use_sparse=False,
                        lamb=self.density*N_e,
                        avoid_self_connections=False,
                        #CHANGE should this be different?
                        eta_stdp = self.eta_stdp)
        tmp = synapses.create_matrix((N_e,self.N),c)
        # get correct connection density
        attempt = 0
        noone = True
        while noone:
            attempt += 1
            mat = (rand(N_e, self.N) < self.density).astype(float)
            nonzeros = int(mat.sum())
            print("[DEBUG] Attempt", attempt, "| Nonzeros:", nonzeros)

            tmp.set_synapses(mat)
            
            if nonzeros > 0:
                noone = False
            if attempt > 1000:
                raise RuntimeError("Stuck in generate_connection_e, failed to create nonzero matrix")

        return tmp
    
    def generate_connection_i(self, N_i):
        # For now, no input to inhibitory units
        c = utils.Bunch(use_sparse=False,
                        lamb=1e-10,
                        avoid_self_connections=False)
        tmp = synapses.create_matrix((N_i, self.N), c)
        tmp.set_synapses(np.zeros((N_i, self.N)))  # No connections
        return tmp
    
class GatedSource(AbstractSource):
    """
    Wraps any input source and suppresses it until a specific timestep.
    """

    def __init__(self, source, gate_start_step):
        """
        Parameters:
            source: instance of an AbstractSource subclass (e.g. RandomSource)
            gate_start_step: int, timestep at which input begins
        """
        self.source = source
        self.gate_start_step = gate_start_step
        self.t = 0
        self.N = source.N  # assumes .N is number of input neurons

    def next(self):
        if self.t < self.gate_start_step:
            self.t += 1
            return np.zeros(self.N)
        else:
            self.t += 1
            return self.source.next()

    def global_index(self):
        return self.source.global_index()

    def global_range(self):
        return self.source.global_range()

    def generate_connection_e(self, N_e):
        return self.source.generate_connection_e(N_e)

    def generate_connection_i(self, N_i):
        return self.source.generate_connection_i(N_i)


# Python 2.7 compatible version for sources.py

class StaticSource(AbstractSource):
    """
    Provides predetermined static input to the network.
    Each excitatory neuron receives a fixed input value throughout the simulation.
    """
    def __init__(self, input_values_e, input_values_i=None, connection_density_e=1.0, 
                 connection_density_i=0.0, eta_stdp=0.0):
        """
        Parameters:
            input_values_e: array-like
                Static input values for excitatory neurons. Shape should be (N_e,) or scalar.
                If scalar, all excitatory neurons receive the same input.
            input_values_i: array-like or None
                Static input values for inhibitory neurons. Shape should be (N_i,) or scalar.
                If None, no input to inhibitory neurons.
            connection_density_e: float
                Fraction of excitatory neurons that receive input (default 1.0 = all)
            connection_density_i: float  
                Fraction of inhibitory neurons that receive input (default 0.0 = none)
            eta_stdp: float
                STDP learning rate for W_eu (default 0.0 = no plasticity)
        """
        self.input_values_e = np.atleast_1d(input_values_e)
        self.input_values_i = np.atleast_1d(input_values_i) if input_values_i is not None else None
        self.connection_density_e = connection_density_e
        self.connection_density_i = connection_density_i
        self.eta_stdp = eta_stdp
        self.N = 1  # We'll use a single "virtual" input neuron
        
    def next(self):
        """
        Returns the static input value.
        Since we're using weight matrices to distribute different values,
        we just return 1.0 to activate our connections.
        """
        return np.array([1.0])
    
    def global_range(self):
        return 1
    
    def global_index(self):
        return 0
    
    def generate_connection_e(self, N_e):
        """
        Generate connection matrix from input to excitatory neurons.
        The weights encode the actual input values.
        """
        # Expand input values to match N_e if needed
        if len(self.input_values_e) == 1:
            # Single value - apply to all neurons
            values = np.ones(N_e) * self.input_values_e[0]
        elif len(self.input_values_e) == N_e:
            # Individual values for each neuron
            values = self.input_values_e.copy()
        else:
            raise ValueError("input_values_e must be scalar or have length N_e")
        
        # Create connection matrix
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,  # Full connectivity
                       avoid_self_connections=False,
                       eta_stdp=self.eta_stdp)
        
        matrix = synapses.create_matrix((N_e, self.N), c)
        
        # Set weights based on input values and connection density
        W = np.zeros((N_e, self.N))
        
        if self.connection_density_e < 1.0:
            # Randomly select which neurons receive input
            connected = np.random.rand(N_e) < self.connection_density_e
            W[connected, 0] = values[connected]
        else:
            # All neurons receive their assigned input
            W[:, 0] = values
            
        matrix.set_synapses(W)
        return matrix
    
    def generate_connection_i(self, N_i):
        """
        Generate connection matrix from input to inhibitory neurons.
        """
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,
                       avoid_self_connections=False)
        
        matrix = synapses.create_matrix((N_i, self.N), c)
        W = np.zeros((N_i, self.N))
        
        if self.input_values_i is not None and self.connection_density_i > 0:
            # Expand input values to match N_i if needed
            if len(self.input_values_i) == 1:
                values = np.ones(N_i) * self.input_values_i[0]
            elif len(self.input_values_i) == N_i:
                values = self.input_values_i.copy()
            else:
                raise ValueError("input_values_i must be scalar or have length N_i")
            
            # Apply connection density
            if self.connection_density_i < 1.0:
                connected = np.random.rand(N_i) < self.connection_density_i
                W[connected, 0] = values[connected]
            else:
                W[:, 0] = values
                
        matrix.set_synapses(W)
        return matrix


class StaticPoissonSource(AbstractSource):
    """
    Alternative implementation that converts static drive values to Poisson rates.
    This maintains the stochastic nature while having predetermined average rates.
    """
    def __init__(self, drive_rates_e, drive_rates_i=None, connection_density_e=1.0,
                 connection_density_i=0.0, eta_stdp=0.0):
        """
        Parameters:
            drive_rates_e: array-like
                Firing rates for each excitatory neuron (between 0 and 1)
            drive_rates_i: array-like or None
                Firing rates for each inhibitory neuron
            connection_density_e/i: float
                Fraction of neurons that receive input
            eta_stdp: float
                STDP learning rate
        """
        self.drive_rates_e = np.atleast_1d(drive_rates_e)
        self.drive_rates_i = np.atleast_1d(drive_rates_i) if drive_rates_i is not None else None
        self.connection_density_e = connection_density_e
        self.connection_density_i = connection_density_i
        self.eta_stdp = eta_stdp
        self.N_e = None  # Will be set when generate_connection_e is called
        self.N_i = None  # Will be set when generate_connection_i is called
        
        # We'll create individual input neurons for each network neuron
        self.N = None  # Will be set based on network size
        
    def next(self):
        """Generate Poisson spikes based on predetermined rates"""
        if self.N is None:
            raise RuntimeError("generate_connection_e/i must be called before next()")
            
        # Generate spikes for all virtual input neurons
        spikes = np.zeros(self.N)
        
        # Excitatory input spikes
        if self.N_e is not None:
            e_rates = self._expand_rates(self.drive_rates_e, self.N_e)
            spikes[:self.N_e] = np.random.rand(self.N_e) < e_rates
            
        # Inhibitory input spikes  
        if self.N_i is not None and self.drive_rates_i is not None:
            i_rates = self._expand_rates(self.drive_rates_i, self.N_i)
            spikes[self.N_e:self.N_e+self.N_i] = np.random.rand(self.N_i) < i_rates
            
        return spikes
    
    def _expand_rates(self, rates, N):
        """Expand rates array to match number of neurons"""
        if len(rates) == 1:
            return np.ones(N) * rates[0]
        elif len(rates) == N:
            return rates.copy()
        else:
            raise ValueError("Rates must be scalar or match number of neurons")
    
    def global_range(self):
        return self.N if self.N is not None else 1
    
    def global_index(self):
        return 0
    
    def generate_connection_e(self, N_e):
        """One-to-one connections from virtual input neurons to excitatory neurons"""
        self.N_e = N_e
        if self.N is None:
            self.N = N_e + (self.N_i if self.N_i is not None else 0)
            
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,
                       avoid_self_connections=False,
                       eta_stdp=self.eta_stdp)
        
        matrix = synapses.create_matrix((N_e, self.N), c)
        
        # One-to-one mapping with connection density
        W = np.zeros((N_e, self.N))
        if self.connection_density_e < 1.0:
            connected = np.random.rand(N_e) < self.connection_density_e
            W[connected, :N_e] = np.eye(N_e)[connected]
        else:
            W[:, :N_e] = np.eye(N_e)
            
        matrix.set_synapses(W)
        return matrix
    
    def generate_connection_i(self, N_i):
        """One-to-one connections from virtual input neurons to inhibitory neurons"""
        self.N_i = N_i
        if self.N is None:
            self.N = (self.N_e if self.N_e is not None else 0) + N_i
        else:
            self.N = self.N_e + N_i
            
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,
                       avoid_self_connections=False)
        
        matrix = synapses.create_matrix((N_i, self.N), c)
        W = np.zeros((N_i, self.N))
        
        if self.drive_rates_i is not None and self.connection_density_i > 0:
            if self.connection_density_i < 1.0:
                connected = np.random.rand(N_i) < self.connection_density_i
                W[connected, self.N_e:] = np.eye(N_i)[connected]
            else:
                W[:, self.N_e:] = np.eye(N_i)
                
        matrix.set_synapses(W)
        return matrix
    
class BurstStaticSource(AbstractSource):
    
    #Provides static input in burst patterns.

    def __init__(self, input_values_e, burst_length=50, gap_length=150,
                 input_values_i=None, connection_density_e=1.0, 
                 connection_density_i=0.0, eta_stdp=0.0):
        """
        Parameters:
            input_values_e: Input strength during bursts
            burst_length: Number of timesteps per burst
            gap_length: Number of timesteps between bursts
            input_values_i: Input to inhibitory neurons (None = no input)
            connection_density_e: Fraction of E neurons receiving input
            connection_density_i: Fraction of I neurons receiving input
            eta_stdp: STDP learning rate
        """
        # Debug: Check what we're receiving
        print "[BurstStaticSource] __init__ called with:"
        print "  input_values_e type:", type(input_values_e)
        print "  input_values_e value:", input_values_e
        if hasattr(input_values_e, 'shape'):
            print "  input_values_e shape:", input_values_e.shape
        if hasattr(input_values_e, 'size'):
            print "  input_values_e size:", input_values_e.size
        
        self.input_values_e = np.atleast_1d(input_values_e)
        self.input_values_i = np.atleast_1d(input_values_i) if input_values_i is not None else None
        self.burst_length = burst_length
        self.gap_length = gap_length
        self.connection_density_e = connection_density_e
        self.connection_density_i = connection_density_i
        self.eta_stdp = eta_stdp
        self.N = 1
        self.t = 0
        
    def next(self):
        """Burst pattern: ON for burst_length, OFF for gap_length"""
        self.t += 1
        cycle_length = self.burst_length + self.gap_length
        phase = self.t % cycle_length
        
        if phase <= self.burst_length:
            return np.array([1.0])  # Burst ON
        else:
            return np.array([0.0])  # Gap
    
    def global_range(self):
        return 1
    
    def global_index(self):
        return 0
    
    def generate_connection_e(self, N_e):
        # Fix: Handle scalar input_values_e properly
        input_array = np.atleast_1d(self.input_values_e)
        
        if input_array.size == 1:
            # Single value - apply to all neurons
            values = np.ones(N_e) * input_array[0]
        elif input_array.size == N_e:
            # Individual values for each neuron
            values = input_array.copy()
        else:
            raise ValueError("input_values_e must be scalar or have length N_e (got length %d, expected %d)" % (input_array.size, N_e))
        
        c = utils.Bunch(use_sparse=False,
                    lamb=np.inf,
                    avoid_self_connections=False,
                    eta_stdp=self.eta_stdp)
        
        matrix = synapses.create_matrix((N_e, self.N), c)
        W = np.zeros((N_e, self.N))
        
        if self.connection_density_e < 1.0:
            connected = np.random.rand(N_e) < self.connection_density_e
            W[connected, 0] = values[connected]
        else:
            W[:, 0] = values
            
        matrix.set_synapses(W)
        return matrix

    def generate_connection_i(self, N_i):
        c = utils.Bunch(use_sparse=False,
                    lamb=np.inf,
                    avoid_self_connections=False)
        
        matrix = synapses.create_matrix((N_i, self.N), c)
        W = np.zeros((N_i, self.N))
        
        if self.input_values_i is not None and self.connection_density_i > 0:
            # Fix: Handle scalar input_values_i properly
            input_array = np.atleast_1d(self.input_values_i)
            
            if input_array.size == 1:
                values = np.ones(N_i) * input_array[0]
            elif input_array.size == N_i:
                values = input_array.copy()
            else:
                raise ValueError("input_values_i must be scalar or have length N_i (got length %d, expected %d)" % (input_array.size, N_i))
            
            if self.connection_density_i < 1.0:
                connected = np.random.rand(N_i) < self.connection_density_i
                W[connected, 0] = values[connected]
            else:
                W[:, 0] = values
                
        matrix.set_synapses(W)
        return matrix

class PoissonBurstSource(AbstractSource):
    """
    Provides static input in burst patterns with stochastic firing.
    """
    def __init__(self, input_values_e, firing_prob=0.4, burst_length=50, gap_length=150,
                 input_values_i=None, connection_density_e=1.0, 
                 connection_density_i=0.0, eta_stdp=0.0):
        """
        Parameters:
            input_values_e: Input strength during bursts
            firing_prob: Probability of firing during burst period (NEW!)
            burst_length: Number of timesteps per burst
            gap_length: Number of timesteps between bursts
            input_values_i: Input to inhibitory neurons (None = no input)
            connection_density_e: Fraction of E neurons receiving input
            connection_density_i: Fraction of I neurons receiving input
            eta_stdp: STDP learning rate
        """
        # Debug: Check what we're receiving
        print "[PoissonBurstSource] __init__ called with:"
        print "  input_values_e type:", type(input_values_e)
        print "  input_values_e value:", input_values_e
        print "  firing_prob:", firing_prob
        if hasattr(input_values_e, 'shape'):
            print "  input_values_e shape:", input_values_e.shape
        if hasattr(input_values_e, 'size'):
            print "  input_values_e size:", input_values_e.size
        
        self.input_values_e = np.atleast_1d(input_values_e)
        self.input_values_i = np.atleast_1d(input_values_i) if input_values_i is not None else None
        self.firing_prob = firing_prob  # NEW!
        self.burst_length = burst_length
        self.gap_length = gap_length
        self.connection_density_e = connection_density_e
        self.connection_density_i = connection_density_i
        self.eta_stdp = eta_stdp
        self.N = 1
        self.t = 0
        
    def next(self):
        """Burst pattern with stochastic firing during bursts"""
        self.t += 1
        cycle_length = self.burst_length + self.gap_length
        phase = self.t % cycle_length
        
        if phase <= self.burst_length:
            # In burst period - use firing_prob
            if np.random.rand() < self.firing_prob:
                return np.array([1.0])  # Fire!
            else:
                return np.array([0.0])  # No fire
        else:
            return np.array([0.0])  # Gap - always silent
    
    def global_range(self):
        return 1
    
    def global_index(self):
        return 0
    
    def generate_connection_e(self, N_e):
        # IDENTICAL to BurstStaticSource
        # Fix: Handle scalar input_values_e properly
        input_array = np.atleast_1d(self.input_values_e)
        
        if input_array.size == 1:
            # Single value - apply to all neurons
            values = np.ones(N_e) * input_array[0]
        elif input_array.size == N_e:
            # Individual values for each neuron
            values = input_array.copy()
        else:
            raise ValueError("input_values_e must be scalar or have length N_e (got length %d, expected %d)" % (input_array.size, N_e))
        
        c = utils.Bunch(use_sparse=False,
                    lamb=np.inf,
                    avoid_self_connections=False,
                    eta_stdp=self.eta_stdp)
        
        matrix = synapses.create_matrix((N_e, self.N), c)
        W = np.zeros((N_e, self.N))
        
        if self.connection_density_e < 1.0:
            connected = np.random.rand(N_e) < self.connection_density_e
            W[connected, 0] = values[connected]
        else:
            W[:, 0] = values
            
        matrix.set_synapses(W)
        return matrix

    def generate_connection_i(self, N_i):
        # IDENTICAL to BurstStaticSource
        c = utils.Bunch(use_sparse=False,
                    lamb=np.inf,
                    avoid_self_connections=False)
        
        matrix = synapses.create_matrix((N_i, self.N), c)
        W = np.zeros((N_i, self.N))
        
        if self.input_values_i is not None and self.connection_density_i > 0:
            # Fix: Handle scalar input_values_i properly
            input_array = np.atleast_1d(self.input_values_i)
            
            if input_array.size == 1:
                values = np.ones(N_i) * input_array[0]
            elif input_array.size == N_i:
                values = input_array.copy()
            else:
                raise ValueError("input_values_i must be scalar or have length N_i (got length %d, expected %d)" % (input_array.size, N_i))
            
            if self.connection_density_i < 1.0:
                connected = np.random.rand(N_i) < self.connection_density_i
                W[connected, 0] = values[connected]
            else:
                W[:, 0] = values
                
        matrix.set_synapses(W)
        return matrix
    
class RandomBurstSource(AbstractSource):
    """
    Random burst source with dynamic neuron selection at each timestep.
    18% chance to select exactly 10% of neurons (which ones varies randomly).
    """
    def __init__(self, input_values_e, firing_prob=0.4, burst_length=50, gap_length=150,
                 input_values_i=None, connection_density_e=1.0, 
                 connection_density_i=0.0, eta_stdp=0.0):
        
        self.input_values_e = np.atleast_1d(input_values_e)
        self.input_values_i = np.atleast_1d(input_values_i) if input_values_i is not None else None
        self.firing_prob = firing_prob
        self.burst_length = burst_length
        self.gap_length = gap_length
        self.connection_density_e = connection_density_e
        self.connection_density_i = connection_density_i
        self.eta_stdp = eta_stdp
        
        # Single input design
        self.N = 1
        self.t = 0
        
    def next(self):
        """Burst pattern with stochastic firing"""
        self.t += 1
        cycle_length = self.burst_length + self.gap_length
        phase = self.t % cycle_length
        
        if phase <= self.burst_length:
            coin_flip = np.random.rand()
            fire = coin_flip < self.firing_prob
            
            # Print coin flip details every 10000 steps or first 10 steps
            if self.t <= 10 or self.t % 10000 == 0:
                print "[RandomBurst] t=%d: coin_flip=%.3f, threshold=%.3f, fire=%s" % (
                    self.t, coin_flip, self.firing_prob, "YES" if fire else "NO")
            
            if fire:
                return np.array([1.0])  # Fire!
            else:
                return np.array([0.0])  # No fire
        else:
            return np.array([0.0])  # Gap - always silent
    
    def global_range(self):
        return 1
    
    def global_index(self):
        return 0
    
    def generate_connection_e(self, N_e):
        """Generate dynamic connection matrix that changes each timestep"""
        input_array = np.atleast_1d(self.input_values_e)
        if input_array.size == 1:
            values = np.ones(N_e) * input_array[0]
        elif input_array.size == N_e:
            values = input_array.copy()
        else:
            raise ValueError("input_values_e must be scalar or have length N_e")
        
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,
                       avoid_self_connections=False,
                       eta_stdp=self.eta_stdp)
        
        # Create the special dynamic matrix
        matrix = DynamicRandomSynapticMatrix((N_e, self.N), c, self, values)
        return matrix
    
    def generate_connection_i(self, N_i):
        """Generate connections for inhibitory neurons (static)"""
        c = utils.Bunch(use_sparse=False,
                       lamb=np.inf,
                       avoid_self_connections=False)
        
        matrix = synapses.create_matrix((N_i, self.N), c)
        matrix.set_synapses(np.zeros((N_i, self.N)))
        return matrix


class DynamicRandomSynapticMatrix(object):
    """
    Matrix that randomly selects exactly 10% of neurons each timestep.
    """
    def __init__(self, shape, c, source, base_values):
        self.shape = shape
        self.c = c
        self.source = source
        self.base_values = base_values.copy()
        self.W = np.zeros(shape)
        
    def __mul__(self, x):
        """Each time this is called, select exactly 10% of neurons randomly"""
        if len(x) > 0 and x[0] > 0:  # Input is active
            # EXACTLY 10% selection each time
            N_e = self.shape[0]
            num_to_select = int(N_e * self.source.connection_density_e)  # Exactly 20 neurons
            
            # Randomly choose exactly num_to_select neurons
            all_neurons = np.arange(N_e)
            selected_indices = np.random.choice(all_neurons, size=num_to_select, replace=False)
            
            # Set connections
            self.W[:, 0] = 0  # Clear all
            self.W[selected_indices, 0] = self.base_values[selected_indices]
            
            # Print selection details when input fires
            # Print every firing event for first 20 steps, then every 10000 steps
            if self.source.t <= 20 or self.source.t % 10000 == 0:
                print "[DynamicMatrix] t=%d: INPUT ACTIVE - Selected EXACTLY %d neurons: %s" % (
                    self.source.t, num_to_select, sorted(selected_indices[:15]))  # Show first 15, sorted
                print "                     Total selected: %d (exactly %.1f%%)" % (
                    num_to_select, 100.0 * num_to_select / N_e)
        else:
            # No input
            self.W[:, 0] = 0
            
            # Occasionally print when no input
            if self.source.t <= 20 or self.source.t % 10000 == 0:
                print "[DynamicMatrix] t=%d: NO INPUT - All neurons silent" % self.source.t
            
        return self.W.dot(x)
    
    # Required methods to mimic synaptic matrix interface
    def stdp(self, *args, **kwargs):
        pass
    
    def set_synapses(self, W):
        self.W = W.copy()
    
    def get_synapses(self):
        return self.W.copy()
    
    def sane_after_update(self):
        return True