from __future__ import division
from pylab import *
from scipy import linalg
import scipy.sparse as sp
import scipy.stats as st

import cPickle as pickle
import gzip

import utils
utils.backup(__file__)
from stats import StatsCollection
from synapses import create_matrix
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import scipy.sparse.linalg as spla

'''
def save_in_chunks(obj, filename, max_bytes=512*1024*1024):  # 1GB chunks
    """Save large object to multiple pickle files"""
    import pickle
    import gzip
    
    bytes_io = pickle.dumps(obj, protocol=2)
    total_bytes = len(bytes_io)
    
    for chunk_id in range(0, total_bytes, max_bytes):
        chunk = bytes_io[chunk_id:chunk_id + max_bytes]
        chunk_name = filename + '.chunk%d' % (chunk_id // max_bytes)
        with gzip.open(chunk_name, 'wb') as f:
            f.write(chunk)

def load_from_chunks(filename_prefix):
    """Load object from chunked pickle files"""
    import pickle
    import gzip
    
    chunks = []
    chunk_id = 0
    while True:
        try:
            chunk_name = filename_prefix + '.chunk%d' % chunk_id
            with gzip.open(chunk_name, 'rb') as f:
                chunks.append(f.read())
            chunk_id += 1
        except IOError:  # No more chunks
            break
    return pickle.loads(''.join(chunks)) 

'''

def save_in_chunks(obj, filename):
    """Save large object by handling numpy arrays and synaptic matrices separately"""
    import pickle
    import numpy as np
    import os
    import types
    from scipy.sparse import csc_matrix
    
    def can_pickle(obj):
        """Test if an object can be pickled"""
        try:
            pickle.dumps(obj, protocol=2)
            return True
        except:
            return False
    
    def _handle_sparse_matrix(matrix):
        """Handle sparse synaptic matrix"""
        data = matrix.W.data.copy() if hasattr(matrix.W, 'data') else None
        indices = matrix.W.indices.copy() if hasattr(matrix.W, 'indices') else None
        indptr = matrix.W.indptr.copy() if hasattr(matrix.W, 'indptr') else None
        return {'data': data, 'indices': indices, 'indptr': indptr}
    
    def _handle_full_matrix(matrix):
        """Handle full synaptic matrix"""
        W = matrix.W.copy() if hasattr(matrix, 'W') else None
        M = matrix.M.copy() if hasattr(matrix, 'M') else None
        return {'W': W, 'M': M}
    
    # Store and remove arrays and matrices
    stored_data = {'arrays': {}, 'matrices': {}, 'objects': {}}
    
    print("\nAnalyzing object structure:")
    for attr_name in dir(obj):
        if attr_name.startswith('__'):
            continue
            
        try:
            attr = getattr(obj, attr_name)
            print(" - %s:" % attr_name, type(attr))
            
            if isinstance(attr, np.ndarray):
                print("   shape:", attr.shape, "dtype:", attr.dtype)
                stored_data['arrays'][attr_name] = attr.copy()
                setattr(obj, attr_name, None)
                
            elif str(type(attr)) == "<class 'common.synapses.SparseSynapticMatrix'>":
                stored_data['matrices'][attr_name] = (_handle_sparse_matrix(attr), 'sparse')
                
            elif str(type(attr)) == "<class 'common.synapses.FullSynapticMatrix'>":
                stored_data['matrices'][attr_name] = (_handle_full_matrix(attr), 'full')
                
            elif not isinstance(attr, (types.MethodType, types.FunctionType)) and can_pickle(attr):
                stored_data['objects'][attr_name] = attr
                
        except Exception as e:
            print("Warning: Could not process attribute %s: %s" % (attr_name, str(e)))
            continue
    
    try:
        # Create arrays directory
        array_dir = filename + '_arrays'
        if not os.path.exists(array_dir):
            os.makedirs(array_dir)
        
        # Save object data
        obj_file = filename + '_objects.pkl'
        print("Saving objects to:", obj_file)
        with open(obj_file, 'wb') as f:
            pickle.dump(stored_data['objects'], f, protocol=2)
        
        # Save numpy arrays
        print("\nSaving arrays and matrices:")
        for arr_name, arr in stored_data['arrays'].items():
            arr_file = os.path.join(array_dir, arr_name + '.npy')
            print(" - Saving array %s with shape %s" % (arr_name, arr.shape))
            np.save(arr_file, arr)
        
        # Save matrix data
        for matrix_name, (matrix_data, matrix_type) in stored_data['matrices'].items():
            for data_name, data in matrix_data.items():
                if data is not None:
                    data_file = os.path.join(array_dir, '%s_%s.npy' % (matrix_name, data_name))
                    print(" - Saving matrix data %s_%s with shape %s" % 
                          (matrix_name, data_name, data.shape))
                    np.save(data_file, data)
                    
    except Exception as e:
        print("Error during save operation:", str(e))
        raise
        
    finally:
        print("\nRestoring object state...")
        # Restore everything
        for attr_name, value in stored_data['objects'].items():
            setattr(obj, attr_name, value)
            
        for arr_name, arr in stored_data['arrays'].items():
            setattr(obj, arr_name, arr)
            
        for matrix_name, (matrix_data, matrix_type) in stored_data['matrices'].items():
            matrix = getattr(obj, matrix_name)
            if matrix_type == 'sparse':
                matrix.W = csc_matrix((matrix_data['data'], 
                                     matrix_data['indices'], 
                                     matrix_data['indptr']))
            else:
                matrix.W = matrix_data['W']
                if 'M' in matrix_data:
                    matrix.M = matrix_data['M']
        
        print("Save operation completed")


def load_from_chunks(filename):
    """Load chunked object"""
    import pickle
    import numpy as np
    import os
    from scipy.sparse import csc_matrix
    
    print("\nStarting load operation...")
    
    try:
        # Create empty object
        class TempObj(object): pass
        obj = TempObj()
        
        # Load pickled objects
        obj_file = filename + '_objects.pkl'
        print("Loading objects from:", obj_file)
        with open(obj_file, 'rb') as f:
            objects = pickle.load(f)
            for key, value in objects.items():
                setattr(obj, key, value)
        
        # Load arrays and matrices
        array_dir = filename + '_arrays'
        print("\nLoading arrays and matrices from:", array_dir)
        
        matrix_data = {}
        for file_name in os.listdir(array_dir):
            if file_name.endswith('.npy'):
                base_name = file_name[:-4]
                data_path = os.path.join(array_dir, file_name)
                print(" - Loading:", file_name)
                
                try:
                    data = np.load(data_path)
                    
                    if '_' in base_name:  # Matrix data
                        matrix_name, data_type = base_name.rsplit('_', 1)
                        if matrix_name not in matrix_data:
                            matrix_data[matrix_name] = {}
                        matrix_data[matrix_name][data_type] = data
                    else:  # Regular array
                        setattr(obj, base_name, data)
                        
                except Exception as e:
                    print("Error loading %s: %s" % (file_name, str(e)))
                    raise
        
        # Reconstruct matrices
        for matrix_name, data in matrix_data.items():
            if 'data' in data:  # Sparse matrix
                matrix = getattr(obj, matrix_name)
                matrix.W = csc_matrix((data['data'], data['indices'], data['indptr']))
            else:  # Full matrix
                matrix = getattr(obj, matrix_name)
                matrix.W = data['W']
                if 'M' in data:
                    matrix.M = data['M']
        
        print("\nLoad operation completed successfully")
        return obj
        
    except Exception as e:
        print("Overall error during load:", str(e))
        raise
    

# Intrinsic Plasticity
def ip(T,x,c):
    """
    Performs intrinsic plasticity
    
    Parameters:
        T: array
            The current thresholds
        x: array
            The state of the network
        c: Bunch
            The parameter bunch
    """
    # T += c.eta_ip*(x.sum()/float(c.N_e)-c.h_ip)
    T += c.eta_ip*(x-c.h_ip)
    return T

class Sorn(object):
    """
    Self-Organizing Recurrent Neural Network
    """
    def __init__(self,c,source):
        """
        Initializes the variables of SORN
        
        Parameters:
            c: bunch
                The bunch of parameters
            source: Source
                The input source
        """
        print("[DEBUG] Entered Sorn.__init__")
        self.c = c 
        self.source = source
        print("[DEBUG] Finished Sorn.__init__")

        # Initialize weight matrices
        # W_to_from (W_ie = from excitatory to inhibitory)
        self.W_ie = create_matrix((c.N_i,c.N_e),c.W_ie)
        self.W_ei = create_matrix((c.N_e,c.N_i),c.W_ei)
        self.W_ee = create_matrix((c.N_e,c.N_e),c.W_ee)
        self.W_eu = self.source.generate_connection_e(c.N_e)
        self.W_iu = self.source.generate_connection_i(c.N_i)

        # print(self.W_eu.W.shape)

        # Initialize the activation of neurons
        self.x = rand(c.N_e)<c.h_ip
        self.y = zeros(c.N_i) # CHANGE_A was rand(c.N_i)<mean(c.h_ip)
        self.u = source.next()

        # Initialize the pre-threshold variables
        self.R_x = zeros(c.N_e)
        self.R_y = zeros(c.N_i)

        # Initialize thresholds
        if c.ordered_thresholds: # Andreeas version
            self.T_i = (arange(c.N_i)+0.5)*((c.T_i_max-c.T_i_min)/(1.*c.N_i))+c.T_i_min## CHANGE_A was c.T_i_min + rand(c.N_i)*(c.T_i_max-c.T_i_min)
            self.T_e = (arange(c.N_e)+0.5)*((c.T_e_max-c.T_e_min)/(1.*c.N_e))+c.T_e_min# CHANGE_A was c.T_e_min + rand(c.N_e)*(c.T_e_max-c.T_e_min)
            shuffle(self.T_e)         # CHANGE_A: only T_e are shuffled
        else:
            self.T_i = c.T_i_min + rand(c.N_i)*(c.T_i_max-c.T_i_min)
            
            #~ maximum thresholds for all neurons
            #~ self.T_e = ones(c.N_e)*c.T_e_max
            #~ minimum thresholds for all neurons
            #~ self.T_e = ones(c.N_e)*c.T_e_min   
                   
            self.T_e = c.T_e_min + rand(c.N_e)*(c.T_e_max-c.T_e_min)

        # Activate plasticity mechanisms
        self.update = True
        self.stats = None

    def step(self,u_new):
        # print("U_new: ", u_new)
        """
        Performs a one-step update of the SORN
        
        Parameters:
            u_new: array
                The input for this step
        """
        c = self.c
        # Compute new state
        self.R_x = self.W_ee*self.x-self.W_ei*self.y-self.T_e
        if not c.noise_sig_e == 0:
            self.R_x += c.noise_sig_e*np.random.randn(c.N_e)
        
        if not c.ff_inhibition_broad == 0:
            self.R_x -= c.ff_inhibition_broad  
             
        x_temp = self.R_x+c.input_gain*(self.W_eu*u_new)

        if c.k_winner_take_all:
            expected = int(round(c.N_e * c.h_ip))
            ind = argsort(x_temp)
            # the next line fails when c.h_ip == 1
            x_new = (x_temp > x_temp[ind[-expected-1]])+0
        else:	
            x_new = (x_temp >= 0.0)+0
            
        # New noise - prob. of each neuron being active 
        if not c.noise_fire == 0:
            x_new += ((np.random.random(c.N_e) < c.noise_fire)+0)
            x_new[x_new > 1] = 1
        # 'Fixed' noise - same neurons are always active
        if c.noise_fire_struc:
            x_new[12:12] = 1 # change number of E neurons for Fig. S2

        if self.c.fast_inhibit:
            x_used = x_new
        else:
            x_used = self.x

        self.R_y = self.W_ie*x_used - self.T_i
        if self.c.ff_inhibition:
            self.R_y += self.W_iu*u_new
        if not c.noise_sig_i == 0:
            self.R_y += c.noise_sig_i*np.random.randn(c.N_i)            
        y_new = (self.R_y >= 0.0)+0

        # Apply plasticity mechanisms
        # Always apply IP
        ip(self.T_e,x_new,self.c)
        # Apply the rest only when update==true
        if self.update:
            assert self.sane_before_update()
            self.W_ee.stdp(self.x,x_new)
            self.W_eu.stdp(self.u,u_new,to_old=self.x,to_new=x_new)

            self.W_ee.struct_p()
            self.W_ei.istdp(self.y,x_new)
            
            self.synaptic_scaling()

            assert self.sane_after_update()

        self.x = x_new
        self.y = y_new
        self.u = u_new
        
        # Update statistics
        self.stats.add()
    
    def synaptic_scaling(self):
        """
        Performs synaptic scaling for all matrices
        """
        self.W_ee.ss()
        self.W_ei.ss() # this was also found in the EM study
        if self.W_eu.c.has_key('eta_stdp') and self.W_eu.c.eta_stdp>0:
            self.W_eu.ss()

    def sane_before_update(self):
        """
        Basic sanity checks for thresholds and states before plasticity
        """
        eps = 1e-6
        assert all(isfinite(self.T_e))
        assert all(isfinite(self.T_i))

        assert all((self.x==0) | (self.x==1))
        assert all((self.y==0) | (self.y==1))

        # TODO quick and easy fix --> this should be 0.0 but threw
        # assertion sometimes on the cluster
        if self.c.noise_sig == -1.0:
            assert all( self.R_x+self.T_e <= +1.0 + eps)
            assert all( self.R_x+self.T_e >= -1.0 - eps)
            assert all( self.R_y+self.T_i <= +1.0 + eps)
            assert all( self.R_y+self.T_i >= 0.0 - eps)

        return True

    def sane_after_update(self):
        """
        Basic sanity checks for matrices after plasticity
        """
        assert self.W_ee.sane_after_update()
        #assert self.W_ei.sane_after_update()
        assert self.W_ie.sane_after_update()

        return True

    def simulation(self,N,toReturn=[]):
        """
        Simulates SORN for a defined number of steps
        
        Parameters:
            N: int
                Simulation steps
            toReturn: list
                Tracking variables to return. Options are: 'X','Y','R_x'
                'R_y', 'U'
        """
        c = self.c
        source = self.source
        ans = {}

        # Initialize tracking variables
        if 'X' in toReturn:
            ans['X'] = zeros( (N,c.N_e) )
        if 'Y' in toReturn:
            ans['Y'] = zeros( (N,c.N_i) )
        if 'R_x' in toReturn:
            ans['R_x'] = zeros( (N,c.N_e) )
        if 'R_y' in toReturn:
            ans['R_y'] = zeros( (N,c.N_i) )
        if 'U' in toReturn:
            ans['U'] = zeros( (N,source.global_range()) )

            # Simulation loop
        for n in range(N):
            if n % 10000 == 0:
                print("[DEBUG] simulation step", n) 

            self.step(source.next())

            #if n % 25000 == 0:
            #    max_eigenvalue = spla.eigs(self.W_ee.W, k=1, return_eigenvectors=False)[0]
            #    max_abs_eigenvalue = np.abs(max_eigenvalue)
            #    print("[INFO] Step %d: Maximum eigenvalue = %.6f" % (n, max_abs_eigenvalue))
            # Simulation step
            # if n % 10000 == 0:
            if False: 
                try:
                    # Try ARPACK first (faster for sparse matrices)
                    max_eigenvalue_E = spla.eigs(self.W_ee.W, k=1, return_eigenvectors=False)[0]
                    max_abs_eigenvalue_E = np.abs(max_eigenvalue_E)
                except ArpackNoConvergence:
                    try:
                        # Fallback 1: Try with increased iterations and tolerance
                        max_eigenvalue_E = spla.eigs(self.W_ee.W, k=1, return_eigenvectors=False, maxiter=5000, tol=1e-8)[0]
                        max_abs_eigenvalue_E = np.abs(max_eigenvalue_E)
                    except:
                        try:
                            # Fallback 2: Use dense matrix eigenvalue computation
                            # Warning: This is more memory intensive
                            max_eigenvalue_E = np.linalg.eigvals(self.W_ee.W.toarray())[0]
                            max_abs_eigenvalue_E = np.abs(max_eigenvalue_E)
                        except:
                            # Final fallback: Use matrix norm as an upper bound
                            print("[WARNING] Eigenvalue computation failed, using spectral norm as approximation")
                            max_abs_eigenvalue_E = np.linalg.norm(self.W_ee.W.toarray(), ord=2)

                # # max_eigenvalues_I = np.linalg.eigvals(self.W_ei.W)
                # # max_abs_eigenvalue_I = np.abs(max_eigenvalues_I)
                #singular_values = np.linalg.svd(self.W_ei.W, compute_uv=False)
                #max_singular_value = np.max(np.abs(singular_values))
                spectral_norm = np.linalg.norm(self.W_ei.W, ord=2)

                activity_x = np.sum(self.x)
                activity_y = np.sum(self.y)

                print("[INFO] Step = {}, activity_x = {}, activity_y = {}, Maximum eigenvalue W_EE = {}, spect norm W_ei = {}".format(n, activity_x, activity_y, max_abs_eigenvalue_E,spectral_norm))

            # Tracking
            if 'X' in toReturn:
                ans['X'][n,:] = self.x
            if 'Y' in toReturn:
                ans['Y'][n,:] = self.y
            if 'R_x' in toReturn:
                ans['R_x'][n,:] = self.R_x
            if 'R_y' in toReturn:
                ans['R_y'][n,:] = self.R_y
            if 'U' in toReturn:
                ans['U'][n,source.global_index()] = 1

            # Command line progress message
            if c.display and (N>100) and \
                    ((n%((N-1)//100) == 0) or (n == N-1)):
                sys.stdout.write('\rSimulation: %3d%%'%((int)(n/(N-1)\
                                                                *100)))
                sys.stdout.flush()
        return ans

    def quicksave(self, filename):
        """Save the network state"""
        try:
            pickle.dump(self, gzip.open(filename, "wb"), protocol=2)
        except OverflowError:
            save_in_chunks(self, filename)

    @staticmethod
    def quickload(filename):
        """Load the network state"""
        try:
            return pickle.load(gzip.open(filename, "rb"))
        except (IOError, EOFError):  # File might be chunked
            return load_from_chunks(filename)


