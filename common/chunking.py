#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
True Chunk File Saving Implementation
=====================================

This modifies the SORN incremental saving to create actual separate chunk files
instead of appending to the same HDF5 file.

Place this file in your common directory and modify your test_single_extrainput_incremental.py
to import and use these classes.
"""

from __future__ import division
import os
import numpy as np
import tables
import utils

class TrueChunkSpikesStat:
    """
    Saves spike data to separate chunk files every N timesteps
    """
    def __init__(self, inhibitory=False, save_interval=10000):
        if inhibitory:
            self.name = 'SpikesInh'
            self.sattr = 'spikes_inh'
        else:
            self.name = 'Spikes'
            self.sattr = 'spikes'
        
        self.collection = 'gather'
        self.inh = inhibitory
        self.save_interval = save_interval
        self.temp_buffer = []
        self.chunk_counter = 0
        self.step = 0
        
    def connect(self, collection):
        """Required method for stats system"""
        pass
        
    def start(self, c, obj):
        """Required method for stats system"""
        pass
        
    def clear(self, c, sorn):
        """Initialize"""
        if self.inh:
            self.neurons = sorn.c.N_i
        else:
            self.neurons = sorn.c.N_e
        
        self.temp_buffer = []
        self.chunk_counter = 0
        self.step = 0
        
        print("[CHUNK] %s: Will save separate files every %d steps" % 
              (self.name, self.save_interval))
        
    def add(self, c, obj):
        """Add spike data to buffer and save as chunk file when full"""
        if self.inh:
            spikes = obj.y
        else:
            spikes = obj.x
            
        # Add to buffer
        self.temp_buffer.append(np.copy(spikes))
        
        # Save chunk if buffer is full
        if len(self.temp_buffer) >= self.save_interval:
            self._save_chunk_file(c, obj)
            
        self.step += 1
        
    def _save_chunk_file(self, c, obj):
        """Save current buffer as a separate chunk file"""
        if not self.temp_buffer:
            return
            
        # Create chunk filename
        chunk_filename = "%s_chunk_%03d.h5" % (self.name, self.chunk_counter)
        chunk_path = utils.logfilename(chunk_filename)
        
        # Convert buffer to array (neurons x timesteps)
        spike_array = np.array(self.temp_buffer).T
        
        # Save to separate HDF5 file
        try:
            with tables.open_file(chunk_path, mode='w') as h5file:
                # Use create_carray for better memory handling
                h5file.create_carray('/', self.name, obj=spike_array)
                
                # Add metadata
                h5file.root._v_attrs.chunk_number = self.chunk_counter
                h5file.root._v_attrs.start_timestep = (self.chunk_counter * self.save_interval)
                h5file.root._v_attrs.end_timestep = ((self.chunk_counter + 1) * self.save_interval - 1)
                h5file.root._v_attrs.n_neurons = self.neurons
                h5file.root._v_attrs.n_timesteps = len(self.temp_buffer)
                
            print("[CHUNK] Saved %s: %s (shape: %s)" % 
                  (chunk_filename, chunk_path, spike_array.shape))
        except Exception as e:
            print("[CHUNK ERROR] Failed to save %s: %s" % (chunk_filename, str(e)))
            # Try alternative save method
            try:
                np.save(chunk_path.replace('.h5', '.npy'), spike_array)
                print("[CHUNK] Saved as NPY instead: %s" % chunk_path.replace('.h5', '.npy'))
            except:
                print("[CHUNK ERROR] Failed to save chunk completely")
        
        # Clear buffer and increment counter
        self.temp_buffer = []
        self.chunk_counter += 1
        
    def finalize(self, c, obj):
        """Save any remaining data in buffer"""
        if self.temp_buffer:
            self._save_chunk_file(c, obj)
            
    def report(self, c, obj):
        """Return empty array as data is in chunk files"""
        self.finalize(c, obj)
        # Return minimal array to satisfy SORN expectations
        return np.zeros((self.neurons, 1))


class ChunkConcatenator:
    """
    Concatenates chunk files back into a single array
    """
    @staticmethod
    def concatenate_chunks(log_path, spike_type='Spikes'):
        """
        Concatenate all chunk files for a given spike type
        
        Args:
            log_path: Path to the simulation directory
            spike_type: 'Spikes' or 'SpikesInh'
            
        Returns:
            Concatenated numpy array (neurons x total_timesteps)
        """
        import glob
        
        # Find all chunk files
        pattern = os.path.join(log_path, "%s_chunk_*.h5" % spike_type)
        chunk_files = sorted(glob.glob(pattern))
        
        if not chunk_files:
            print("No chunk files found for %s" % spike_type)
            return None
            
        print("Found %d chunk files for %s" % (len(chunk_files), spike_type))
        
        # Read and concatenate
        all_chunks = []
        total_timesteps = 0
        
        for chunk_file in chunk_files:
            with tables.open_file(chunk_file, mode='r') as h5file:
                # Read chunk data
                if hasattr(h5file.root, spike_type):
                    chunk_data = h5file.root[spike_type].read()
                else:
                    # Try alternative names
                    for node in h5file.root:
                        if spike_type in node._v_name:
                            chunk_data = node.read()
                            break
                
                all_chunks.append(chunk_data)
                total_timesteps += chunk_data.shape[1]
                
                print("  Loaded chunk %d: shape %s" % 
                      (h5file.root._v_attrs.chunk_number, chunk_data.shape))
        
        # Concatenate along time axis
        concatenated = np.concatenate(all_chunks, axis=1)
        print("Concatenated shape: %s" % str(concatenated.shape))
        
        return concatenated
    
    @staticmethod
    def save_concatenated(log_path, output_filename='concatenated_spikes.h5'):
        """
        Concatenate all chunks and save to a single file
        """
        output_path = os.path.join(log_path, output_filename)
        
        with tables.open_file(output_path, mode='w') as h5file:
            # Concatenate excitatory spikes
            spikes = ChunkConcatenator.concatenate_chunks(log_path, 'Spikes')
            if spikes is not None:
                h5file.create_array('/', 'Spikes', spikes)
                
            # Concatenate inhibitory spikes
            spikes_inh = ChunkConcatenator.concatenate_chunks(log_path, 'SpikesInh')
            if spikes_inh is not None:
                h5file.create_array('/', 'SpikesInh', spikes_inh)
                
        print("Saved concatenated data to: %s" % output_path)
        return output_path


def modify_sorn_for_true_chunk_saving(sorn, save_interval=10000):
    """
    Modify SORN to use true chunk file saving
    
    Usage in your test_single_extrainput_incremental.py:
        from true_chunk_saving import modify_sorn_for_true_chunk_saving
        sorn = modify_sorn_for_true_chunk_saving(sorn, save_interval=10000)
    """
    print("\n" + "="*50)
    print("ENABLING TRUE CHUNK FILE SAVING")
    print("="*50)
    
    # Find and replace spike stats
    new_methods = []
    replaced_e = False
    replaced_i = False
    
    for method in sorn.stats.methods:
        # Replace SpikesStat with chunk version
        if hasattr(method, 'name'):
            if method.name == 'Spikes' and not replaced_e:
                new_method = TrueChunkSpikesStat(inhibitory=False, save_interval=save_interval)
                new_methods.append(new_method)
                replaced_e = True
                print("Replaced Spikes stat with chunk saving version")
            elif method.name == 'SpikesInh' and not replaced_i:
                new_method = TrueChunkSpikesStat(inhibitory=True, save_interval=save_interval)
                new_methods.append(new_method)
                replaced_i = True
                print("Replaced SpikesInh stat with chunk saving version")
            else:
                new_methods.append(method)
        else:
            new_methods.append(method)
    
    # If spike stats weren't found, add them
    if not replaced_e:
        new_method = TrueChunkSpikesStat(inhibitory=False, save_interval=save_interval)
        new_methods.append(new_method)
        print("Added chunk saving for excitatory spikes")
        
    if not replaced_i:
        new_method = TrueChunkSpikesStat(inhibitory=True, save_interval=save_interval)
        new_methods.append(new_method)
        print("Added chunk saving for inhibitory spikes")
    
    sorn.stats.methods = new_methods
    
    # Clear and restart stats
    sorn.stats.clear()
    
    print("Chunk saving enabled: files will be saved every %d timesteps" % save_interval)
    print("="*50 + "\n")
    
    return sorn


# Example usage for direct import
def concatenate_simulation_chunks(log_path, delete_chunks=True):
    """
    Concatenate chunks for a simulation
    
    Args:
        log_path: Path to simulation common directory
        delete_chunks: Whether to delete chunk files after concatenation
        
    Returns:
        Path to concatenated file or None
    """
    concatenator = ChunkConcatenator()
    output_file = concatenator.save_concatenated(log_path)
    
    if output_file and delete_chunks:
        # Delete chunk files
        import glob
        for pattern in ['Spikes_chunk_*.h5', 'SpikesInh_chunk_*.h5']:
            for chunk_file in glob.glob(os.path.join(log_path, pattern)):
                try:
                    os.remove(chunk_file)
                    print("Deleted: %s" % chunk_file)
                except:
                    pass
                    
    return output_file