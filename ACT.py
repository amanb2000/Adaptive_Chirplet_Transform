"""
===========================================
=== Adaptive Chirplet Transform Library ===
===========================================

# Core Functions 

1. g(): Function for generating chirplets. 
2. generate_chirplet_dictionary(): Generates a dictionary of chirplets along with a matrix of the associated chirplet parameters.
3. search_dictionary(): Searches for the entry in the dictionary that has the greatest projection value with a given signal.
4. transform(): Performs a P-Order ACT approximation of an input signal using the configured dictionary.
5. minimize_this(): Cost function for the goodness of coefficient set given a signal.

# Argument Parameters

1. FS (in hertz),           DEFAULT: 128
2. length (in samples),     DEFAULT: 76
3. tc_info, fc_info, logDt_info, c_info: tuples (min, max, step) for generating chirplet dictionary, DEFAULT: 
    tc_info     = (0, 76, 1)
    fc_info     = (.5, 15, .2)
    logDt_info  = (-4, -1, .3) 
    c_info      = (-30, 30, 1)

# Computed Parameters

1. dict_mat
2. param_mat
"""

import os
import pickle
import numpy as np
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import scipy.optimize as optimize
from tqdm import tqdm
import joblib

class ACT:
    def __init__(self, FS=128, length=76, dict_addr='dict_cache.p', tc_info=(0, 76, 1), fc_info=(.7, 15, .2), 
            logDt_info=(-4, -1, .3), c_info=(-30, 30, 3), complex=False, force_regenerate=False, mute=False):

        # Adding immediately presented arguments.
        self.FS =           FS
        self.length =       length

        self.dict_addr =    dict_addr

        self.tc_info =      tc_info 
        self.fc_info =      fc_info 
        self.logDt_info =   logDt_info 
        self.c_info =       c_info 

        self.complex =      False
        self.float32 =      True

        if not mute:
            print('\n===============================================')
            print('INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE')
            print('===============================================\n\n')

        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute: print("Found Chirplet Dictionary, Loading File...")

            self.dict_mat, self.param_mat = joblib.load(self.dict_addr)

        else:
            if not mute: print("Did not find cached chirplet dictionary matrix, generating chirplet dictionary...\n")
            self.generate_chirplet_dictionary(debug=True)
            if not mute: print("\nDone Generating Chirplet Dictionary")

            if not mute: print("\nCaching Generated Dictionary/Parameter Matrices...")

            joblib.dump( (self.dict_mat, self.param_mat), self.dict_addr )
            if not mute: print("Done Caching.")

        if not mute: 
            print("=====================================================")
            print("DONE INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE.")
            print("=====================================================")

        
    def g(self, tc=0, fc=1, logDt=0, c=0):
        """
        Function for creating a numpy array of a Gaussian Chirplet:
        
        tc: Time center (in SAMPLES)
        fc: Frequency center (in HERTZ)
        logDt: Log of Delta_t, the length of the chirplet (0 == ONE SECOND LONG)
        c: Chirp rate (in HERTZ/sec)
        
        FS: Frequency of the signal
        length: Desired length for the chirplet (NUMBER OF SAMPLES)
        """

        tc /= self.FS # Converting back to seconds for the sake of the synthesis
        
        Dt = np.exp(logDt) # Calculating the non-log Delta_t value
        t = np.arange(self.length)/self.FS # Time array
        
        gaussian_window = np.exp(-0.5 * ((t - tc)/(Dt))**2)
        
        complex_exp = np.exp(2j*np.pi * (c*(t-tc)**2 + fc*(t-tc)) )
        
        final_chirplet = gaussian_window*complex_exp
        
        if not self.complex:
            final_chirplet = np.real(final_chirplet)

        if self.float32:
            final_chirplet = final_chirplet.astype(np.float32)

        return final_chirplet

    def generate_chirplet_dictionary(self, debug=False):
        """
        Function for creating a chirplet dictionary.

        TAKES:
        - tc_info:      (Innate) Tuple of values pertaining to time centre: (min, max, step_size)
        - fc_info:      (Innate) Tuple of values pertaining to frequency centre: (min, max, step_size)
        - logDt_info:   (Innate) Tuple of values pertaining to time duration: (min, max, step_size)
        - c_info:       (Innate) Tuple of values pertaining to chirprate: (min, max, step_size)
        - length:       (Innate) Desired NUMBER OF SAMPLES in each chirplet.
        - FS:           (Innate) Sampling frequency (defaults to 128 for initial EEG paper analyses).

        RETURNS:
        - dict_mat:     2-D matrix representing the chirplet dictionary. [[chirp_0], ..., [chirp_n]]
        - param_mat:    2-D matrix with the corresponding tuples of parameters. [[tc_0, fc_0, logDt_0, c_0], ..., [tc_n, fc_n, logDt_n, c_n]]
        """

        tc_vals = np.arange(self.tc_info[0], self.tc_info[1], self.tc_info[2])
        fc_vals = np.arange(self.fc_info[0], self.fc_info[1], self.fc_info[2])
        logDt_vals = np.arange(self.logDt_info[0], self.logDt_info[1], self.logDt_info[2])
        c_vals = np.arange(self.c_info[0], self.c_info[1], self.c_info[2])
        
        dict_size = int(len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals))
        
        if debug:
            print("Dictionary length: {}".format(dict_size))

        dict_mat = np.zeros( [dict_size, self.length] )
        param_mat = np.zeros( [dict_size, 4 ] ) # 4 dimension parameter space

        cnt = 0
        slow_cnt = 1 # For debugging purposes (pretty progress marker)
        for tc in tc_vals:
            if debug:
                print('\n{}/{}: \t'.format(slow_cnt, len(tc_vals)), end='')
                slow_cnt += 1
            for fc in fc_vals:
                if debug:
                    print('.', end='')
                for logDt in logDt_vals:
                    for c in c_vals:
                        dict_mat[cnt] = self.g(tc=tc, fc=fc, logDt=logDt, c=c)
                        param_mat[cnt] = np.asarray([tc, fc, logDt, c])

                        cnt += 1

        self.dict_mat = dict_mat.astype(np.float32)
        self.param_mat = param_mat.astype(np.float32)

        return dict_mat, param_mat
        


    def search_dictionary(self, signal):
        """
        TODO: Populate this function description
        """
        return np.argmax(self.dict_mat.dot(signal)), np.max(self.dict_mat.dot(signal))

    def transform(self, signal, order=5, debug=False):

        param_list = np.zeros( [order, 4] ) # order x 4 matrix of the 4 parameters corresponding to each of the 
                                            # approximation chirplets.
        coeff_list = np.zeros(order)        # Associated coefficients for each of the 4-long parameter sets. 

        approx = np.zeros(len(signal))

        residue = np.copy(signal)

        if debug:
            print('Beginning {}-Order Transform of Input Signal...'.format(order))
        for P in range(order):
            if debug: 
                print(".", end="")
            
            # Generating the index and inner product of the MP-dictated chirplet from the dictionary
            ind, val = self.search_dictionary(residue)

            params = self.param_mat[ind] # Coarse estimation parameters (np array)

            res = optimize.minimize(self.minimize_this, params, args=(residue)) # Fine-tuning using BFGS
            new_params = res.x

            # print('\n\nOptimizer Exit Code (0 = Good): {}'.format(res.status))

            if res.status != 0 and debug: 
                print('OPTIMIZER DID NOT TERMINATE SUCCESSFULLY!!!')
                print('Message: {}'.format(res.message))
            
            updated_base_chirplet = self.g(tc=new_params[0], fc=new_params[1], logDt=new_params[2], c=new_params[3])
            updated_chirplet_coeff = updated_base_chirplet.dot(signal)/self.FS

            new_chirp = updated_base_chirplet * updated_chirplet_coeff

            # Updating the signal and the current approximation
            residue -= new_chirp
            approx += new_chirp

            param_list[P] = new_params
            coeff_list[P] = updated_chirplet_coeff

        if debug:
            print('')

        return {
            'params': param_list,
            'coeffs': coeff_list,
            'signal': signal,
            'error': np.sum(residue),
            'residue': residue,
            'approx': approx
        }


    def minimize_this(self, coeffs, signal):
        """
        Function to be minimized in the fine-tuning steps.

        TODO: Fill out this function description.
        """

        atom = self.g(tc=coeffs[0], fc=coeffs[1], logDt=coeffs[2], c=coeffs[3])

        return -1*abs(atom.dot(signal))
