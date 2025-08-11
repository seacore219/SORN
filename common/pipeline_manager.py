#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows-RIS Data Pipeline Manager for SORN Simulations
======================================================

This script manages data transfer from RIS (remote) to Windows local machine.
Adapted for:
- Local: C:\Users\seaco\OneDrive\Documents\Charles\SORN_RIS\backup\test_single
- Remote: c.dumoulin@compute1-client-1.ris.wustl.edu

Usage:
    python data_pipeline_manager_windows.py
"""

from __future__ import division, print_function
import os
import sys
import time
import hashlib
import json
import subprocess
import numpy as np
import tables
from datetime import datetime
import logging
import platform

class WindowsRISPipelineManager:
    """Manages data transfer between RIS and Windows local machine"""
    
    def __init__(self):
        # Windows local path
        self.local_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_RIS\backup"
        
        # RIS connection info
        self.ris_user = "c.dumoulin"
        self.ris_host = "compute1-exec-71"
        self.ris_backup_path = "/home/c.dumoulin/SORN_RIS/backup"
        
        # Pipeline parameters
        self.chunk_size = 10000
        self.cleanup_interval = 30000  # Delete after 30k timesteps
        self.chunks_to_keep = 3  # Always keep last 3 chunks as safety buffer
        
        # Setup
        self.setup_logging()
        self.manifest_file = os.path.join(self.local_path, "chunk_manifest.json")
        self.load_manifest()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.local_path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, "pipeline_manager.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_manifest(self):
        """Load chunk manifest from file"""
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as f:
                self.chunk_manifest = json.load(f)
        else:
            self.chunk_manifest = {}
            
    def save_manifest(self):
        """Save chunk manifest to file"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.chunk_manifest, f, indent=2)
            
    def compute_file_hash(self, filepath):
        """Compute MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def run_ris_command(self, command):
        """Run command on RIS via SSH"""
        ssh_command = [
            "ssh",
            "%s@%s" % (self.ris_user, self.ris_host),
            command
        ]
        
        try:
            result = subprocess.check_output(ssh_command, universal_newlines=True)
            return result.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error("RIS command failed: %s", e)
            return None
            
    def scp_from_ris(self, remote_file, local_file):
        """Copy file from RIS to local using SCP"""
        # Create local directory if needed
        local_dir = os.path.dirname(local_file)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            
        scp_command = [
            "scp",
            "%s@%s:%s" % (self.ris_user, self.ris_host, remote_file),
            local_file
        ]
        
        try:
            subprocess.check_call(scp_command)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error("SCP failed: %s", e)
            return False
            
    def scan_ris_chunks(self, simulation_id):
        """Scan RIS for available chunks"""
        remote_sim_path = os.path.join(self.ris_backup_path, simulation_id)
        
        # List files on RIS
        command = "ls -la %s/*_chunk_*.h5 2>/dev/null || true" % remote_sim_path
        result = self.run_ris_command(command)
        
        if not result:
            return []
            
        chunks = []
        for line in result.split('\n'):
            if '_chunk_' in line and line.endswith('.h5'):
                parts = line.split()
                if len(parts) >= 9:
                    filename = parts[-1].split('/')[-1]
                    size = int(parts[4])
                    
                    # Extract chunk number
                    chunk_num = int(filename.split('_chunk_')[1].split('.')[0])
                    
                    chunks.append({
                        'filename': filename,
                        'filepath': os.path.join(remote_sim_path, filename),
                        'chunk_num': chunk_num,
                        'size': size
                    })
                    
        return sorted(chunks, key=lambda x: x['chunk_num'])
        
    def transfer_chunk(self, chunk_info, simulation_id):
        """Transfer a single chunk from RIS to local"""
        local_sim_path = os.path.join(self.local_path, simulation_id, 'chunks')
        if not os.path.exists(local_sim_path):
            os.makedirs(local_sim_path)
            
        local_chunk_path = os.path.join(local_sim_path, chunk_info['filename'])
        
        # Check if already transferred
        manifest_key = "%s/%s" % (simulation_id, chunk_info['filename'])
        if os.path.exists(local_chunk_path) and manifest_key in self.chunk_manifest:
            self.logger.info("Chunk already transferred: %s", chunk_info['filename'])
            return True
            
        # Transfer the file
        self.logger.info("Transferring chunk: %s", chunk_info['filename'])
        if self.scp_from_ris(chunk_info['filepath'], local_chunk_path):
            # Verify transfer by checking file size
            local_size = os.path.getsize(local_chunk_path)
            if local_size != chunk_info['size']:
                self.logger.error("Size mismatch for chunk: %s", chunk_info['filename'])
                os.remove(local_chunk_path)
                return False
                
            # Update manifest
            self.chunk_manifest[manifest_key] = {
                'size': chunk_info['size'],
                'chunk_num': chunk_info['chunk_num'],
                'transferred_at': time.time()
            }
            self.save_manifest()
            
            self.logger.info("Successfully transferred: %s", chunk_info['filename'])
            return True
        else:
            return False
            
    def notify_ris_cleanup(self, simulation_id, chunks_to_delete):
        """Create a cleanup script on RIS for this simulation"""
        if not chunks_to_delete:
            return
            
        # Create cleanup commands
        cleanup_script = "#!/bin/bash\n"
        cleanup_script += "# Auto-generated cleanup script\n"
        cleanup_script += "echo 'Cleaning up transferred chunks for %s'\n" % simulation_id
        
        for chunk in chunks_to_delete:
            cleanup_script += "rm -f %s\n" % chunk['filepath']
            cleanup_script += "echo 'Deleted: %s'\n" % chunk['filename']
            
        # Write script to RIS
        script_name = "cleanup_%s_%s.sh" % (simulation_id, datetime.now().strftime("%Y%m%d_%H%M%S"))
        remote_script_path = "/tmp/%s" % script_name
        
        # Create script on RIS
        command = "cat > %s << 'EOF'\n%sEOF" % (remote_script_path, cleanup_script)
        self.run_ris_command(command)
        
        # Make executable
        self.run_ris_command("chmod +x %s" % remote_script_path)
        
        self.logger.info("Created cleanup script on RIS: %s", remote_script_path)
        self.logger.info("Run it with: ssh %s@%s 'bash %s'" % 
                        (self.ris_user, self.ris_host, remote_script_path))
                        
    def concatenate_chunks(self, simulation_id):
        """Concatenate chunks for a simulation"""
        local_sim_path = os.path.join(self.local_path, simulation_id, 'chunks')
        
        if not os.path.exists(local_sim_path):
            self.logger.error("No chunks found for simulation: %s", simulation_id)
            return False
            
        # Use the local concatenator logic
        output_path = os.path.join(self.local_path, simulation_id, "concatenated_result.h5")
        
        # Find all chunk files
        spike_chunks = []
        spike_inh_chunks = []
        
        for filename in os.listdir(local_sim_path):
            if not filename.endswith('.h5'):
                continue
                
            filepath = os.path.join(local_sim_path, filename)
            
            if 'SpikesInh_chunk_' in filename:
                chunk_num = int(filename.split('_chunk_')[1].split('.')[0])
                spike_inh_chunks.append((chunk_num, filepath))
            elif 'Spikes_chunk_' in filename:
                chunk_num = int(filename.split('_chunk_')[1].split('.')[0])
                spike_chunks.append((chunk_num, filepath))
                
        # Sort by chunk number
        spike_chunks.sort(key=lambda x: x[0])
        spike_inh_chunks.sort(key=lambda x: x[0])
        
        self.logger.info("Concatenating %d Spikes chunks and %d SpikesInh chunks", 
                        len(spike_chunks), len(spike_inh_chunks))
                        
        try:
            with tables.open_file(output_path, mode='w') as h5file:
                # Concatenate excitatory spikes
                if spike_chunks:
                    all_spikes = []
                    for chunk_num, chunk_path in spike_chunks:
                        with tables.open_file(chunk_path, mode='r') as chunk_file:
                            for node in chunk_file.walk_nodes('/', classname='Array'):
                                if 'Spikes' in node._v_name and 'Inh' not in node._v_name:
                                    all_spikes.append(node.read())
                                    
                    if all_spikes:
                        concatenated = np.concatenate(all_spikes, axis=1)
                        h5file.create_array('/', 'Spikes', concatenated)
                        self.logger.info("Saved Spikes: shape %s", concatenated.shape)
                        
                # Concatenate inhibitory spikes
                if spike_inh_chunks:
                    all_spikes_inh = []
                    for chunk_num, chunk_path in spike_inh_chunks:
                        with tables.open_file(chunk_path, mode='r') as chunk_file:
                            for node in chunk_file.walk_nodes('/', classname='Array'):
                                if 'SpikesInh' in node._v_name:
                                    all_spikes_inh.append(node.read())
                                    
                    if all_spikes_inh:
                        concatenated_inh = np.concatenate(all_spikes_inh, axis=1)
                        h5file.create_array('/', 'SpikesInh', concatenated_inh)
                        self.logger.info("Saved SpikesInh: shape %s", concatenated_inh.shape)
                        
            self.logger.info("Concatenation complete: %s", output_path)
            return True
            
        except Exception as e:
            self.logger.error("Error concatenating chunks: %s", str(e))
            return False
            
    def process_simulation(self, simulation_id):
        """Process a single simulation"""
        self.logger.info("Processing simulation: %s", simulation_id)
        
        # 1. Scan for chunks on RIS
        chunks = self.scan_ris_chunks(simulation_id)
        self.logger.info("Found %d chunks on RIS", len(chunks))
        
        if not chunks:
            return
            
        # 2. Transfer new chunks
        transferred = 0
        for chunk in chunks:
            if self.transfer_chunk(chunk, simulation_id):
                transferred += 1
                
        self.logger.info("Transferred %d new chunks", transferred)
        
        # 3. Check if we should trigger cleanup
        # Get the highest chunk number
        max_chunk_num = max(chunk['chunk_num'] for chunk in chunks)
        max_timestep = (max_chunk_num + 1) * self.chunk_size
        
        # Cleanup if we've passed the cleanup interval
        if max_timestep >= self.cleanup_interval:
            # Determine which chunks to delete (keep last 3)
            chunks_to_delete = []
            
            for chunk in chunks[:-self.chunks_to_keep]:  # Keep last 3 chunks
                manifest_key = "%s/%s" % (simulation_id, chunk['filename'])
                # Only delete if confirmed transferred
                if manifest_key in self.chunk_manifest:
                    chunks_to_delete.append(chunk)
                    
            if chunks_to_delete:
                self.logger.info("Ready to delete %d chunks (keeping last %d)", 
                               len(chunks_to_delete), self.chunks_to_keep)
                               
                # Concatenate before cleanup
                if self.concatenate_chunks(simulation_id):
                    # Create cleanup script on RIS
                    self.notify_ris_cleanup(simulation_id, chunks_to_delete)
                else:
                    self.logger.error("Concatenation failed, skipping cleanup")
                    
    def list_ris_simulations(self):
        """List all simulations on RIS"""
        command = "ls -d %s/*/ 2>/dev/null || true" % self.ris_backup_path
        result = self.run_ris_command(command)
        
        if not result:
            return []
            
        simulations = []
        for line in result.split('\n'):
            if line.strip():
                sim_name = line.strip().rstrip('/').split('/')[-1]
                if sim_name:
                    simulations.append(sim_name)
                    
        return simulations
        
    def monitor_loop(self, check_interval=60):
        """Main monitoring loop"""
        self.logger.info("Starting Windows-RIS pipeline monitor...")
        self.logger.info("Local path: %s", self.local_path)
        self.logger.info("RIS: %s@%s:%s", self.ris_user, self.ris_host, self.ris_backup_path)
        self.logger.info("Check interval: %d seconds", check_interval)
        self.logger.info("Cleanup after: %d timesteps", self.cleanup_interval)
        self.logger.info("Chunks to keep: %d", self.chunks_to_keep)
        
        while True:
            try:
                # Get list of simulations on RIS
                simulations = self.list_ris_simulations()
                self.logger.info("Found %d simulations on RIS", len(simulations))
                
                # Process each simulation
                for sim_id in simulations:
                    self.process_simulation(sim_id)
                    
                # Wait before next check
                self.logger.info("Waiting %d seconds before next check...", check_interval)
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Monitor stopped by user")
                break
            except Exception as e:
                self.logger.error("Error in monitor loop: %s", str(e))
                time.sleep(check_interval)


def main():
    print("="*60)
    print("Windows-RIS Data Pipeline Manager")
    print("="*60)
    
    # Check if on Windows
    if platform.system() != 'Windows':
        print("WARNING: This script is designed for Windows.")
        
    print("\nConfiguration:")
    print("Local path: C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\SORN_RIS\\backup")
    print("RIS host: c.dumoulin@compute1-exec-71")
    print("RIS path: /home/c.dumoulin/SORN_RIS/backup")
    print("\nPress Enter to start or Ctrl+C to exit...")
    input()
    
    # Create and run manager
    manager = WindowsRISPipelineManager()
    
    print("\nStarting pipeline manager...")
    print("Press Ctrl+C to stop")
    
    manager.monitor_loop(check_interval=60)=60)


if __name__ == "__main__":
    main()