#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel batch runner for SORN simulations
Keeps the simulation mechanics identical, just runs them in parallel
"""

from __future__ import division, print_function
import os
import sys
import subprocess
import multiprocessing
import time
import argparse
import shutil
import datetime

def find_python27():
    """Find Python 2.7 executable"""
    for cmd in ["python2.7", "python2", "python27.exe", "python27", "python"]:
        try:
            result = subprocess.check_output([cmd, "--version"], 
                                           stderr=subprocess.STDOUT, 
                                           universal_newlines=True)
            if "Python 2.7" in result:
                return cmd
        except:
            continue
    
    # Windows specific paths
    if os.name == 'nt':
        common_paths = [
            "C:\\Python27\\python.exe",
            "C:\\Python27\\python27.exe",
            "C:\\Python\\Python27\\python.exe",
            "C:\\Python\\Python27\\python27.exe"
        ]
        for win_path in common_paths:
            if os.path.exists(win_path):
                return win_path
    
    raise Exception("Cannot find Python 2.7")

def get_batch_folder_name(param_file):
    """
    Create batch folder name from parameters
    Reads h_ip, firing_prob, connection_density_e, connection_density_i from param file
    """
    try:
        # Read the parameter file and extract values
        # We'll parse it manually to avoid import issues
        params = {}
        
        # Read the file and look for the parameters we need
        with open(param_file, 'r') as f:
            content = f.read()
            
        # Extract h_ip
        if 'c.h_ip' in content:
            h_ip_line = [line for line in content.split('\n') if 'c.h_ip' in line and '=' in line]
            if h_ip_line:
                h_ip = float(h_ip_line[0].split('=')[1].strip())
            else:
                h_ip = 0.0075  # default
        else:
            h_ip = 0.0075
            
        # Look for perturbation_source parameters
        firing_prob = 0.01  # default
        conn_density_e = 0.01  # default
        conn_density_i = 0.01  # default
        
        if 'firing_prob=' in content:
            fp_line = [line for line in content.split('\n') if 'firing_prob=' in line]
            if fp_line:
                # Extract the number from firing_prob=0.01,
                fp_str = fp_line[0].split('firing_prob=')[1].split(',')[0].strip()
                firing_prob = float(fp_str)
        
        if 'connection_density_e=' in content:
            cde_line = [line for line in content.split('\n') if 'connection_density_e=' in line]
            if cde_line:
                cde_str = cde_line[0].split('connection_density_e=')[1].split(',')[0].strip()
                conn_density_e = float(cde_str)
                
        if 'connection_density_i=' in content:
            cdi_line = [line for line in content.split('\n') if 'connection_density_i=' in line]
            if cdi_line:
                cdi_str = cdi_line[0].split('connection_density_i=')[1].split(',')[0].strip()
                conn_density_i = float(cdi_str)

        if 'noise_sig_e=' in content:
            nse_line = [line for line in content.split('\n') if 'noise_sig_e=' in line]
            if nse_line:
                nse_str = nse_line[0].split('noise=')[1].split(',')[0].strip()
                noise_sig_e = float(nse_str)

        # Create folder name with parameters only (no timestamp)
        folder_name = "batch_hip%.4f_fp%.2f_cde%.2f_cdi%.2f_noise%.2f" % (
            h_ip, firing_prob, conn_density_e, conn_density_i, noise_sig_e
        )

        print("Extracted parameters: h_ip=%.4f, fp=%.2f, cde=%.2f, cdi=%.2f, noise=%.2f" % 
              (h_ip, firing_prob, conn_density_e, conn_density_i, noise_sig_e))

        return folder_name
        
    except Exception as e:
        print("Warning: Could not read parameters for folder name: %s" % str(e))
        # Fallback to timestamp only
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return "batch_%s" % timestamp

def organize_simulation_results(batch_folder, num_sims, root_dir):
    """
    Move simulation results from backup folders to organized batch folder
    """
    # Use the specific backup directory path
    backup_dir = os.path.join(os.path.dirname(root_dir), "backup")
    batch_path = os.path.join(backup_dir, batch_folder)
    
    print("\nOrganizing results...")
    print("Looking in backup directory: %s" % backup_dir)
    print("Will create batch folder: %s" % batch_path)
    
    # Create batch folder if it doesn't exist
    if not os.path.exists(batch_path):
        try:
            os.makedirs(batch_path)
            print("Created batch folder: %s" % batch_path)
        except Exception as e:
            print("ERROR: Could not create batch folder: %s" % str(e))
            return None
    
    # Find and move simulation folders
    moved_count = 0
    test_single_dir = os.path.join(backup_dir, "test_single")
    
    print("Looking for simulation folders in: %s" % test_single_dir)
    
    if not os.path.exists(test_single_dir):
        print("ERROR: test_single directory not found at: %s" % test_single_dir)
        return batch_path
    
    # Get all folders sorted by creation time (most recent first)
    all_folders = []
    try:
        for folder in os.listdir(test_single_dir):
            folder_path = os.path.join(test_single_dir, folder)
            if os.path.isdir(folder_path):
                # Get creation time
                ctime = os.path.getctime(folder_path)
                all_folders.append((ctime, folder, folder_path))
        
        print("Found %d folders in test_single" % len(all_folders))
        
        # Sort by creation time (newest first)
        all_folders.sort(reverse=True)
        
        # Move the most recent num_sims folders
        for i, (ctime, folder_name, folder_path) in enumerate(all_folders[:num_sims]):
            sim_number = i + 1
            new_name = "sim_%02d_%s" % (sim_number, folder_name)
            dest_path = os.path.join(batch_path, new_name)
            
            try:
                print("Moving: %s -> %s" % (folder_name, new_name))
                shutil.move(folder_path, dest_path)
                moved_count += 1
                print("  Moved simulation %d successfully" % sim_number)
            except Exception as e:
                print("  WARNING: Could not move %s: %s" % (folder_name, str(e)))
                
    except Exception as e:
        print("ERROR while organizing: %s" % str(e))
    
    print("\nMoved %d simulation folders to: %s" % (moved_count, batch_folder))
    return batch_path

def run_single_simulation(args):
    """
    Run a single simulation with its unique number
    This function is called by each parallel process
    """
    sim_number, python_exe, test_single, param_file, common_dir, show_output = args

    # Stagger the start times by 1 second per simulation to avoid folder conflicts
    stagger_delay = (sim_number - 1) * 1.0  # 1 second per simulation
    if stagger_delay > 0:
        print("[Sim %d] Waiting %.1f seconds before starting..." % (sim_number, stagger_delay))
        time.sleep(stagger_delay)
    
    # Set up environment with simulation number
    env = dict(os.environ)
    env['SORN_SIM_NUMBER'] = str(sim_number)
    
    # Add root directory to PYTHONPATH
    root_dir = os.path.dirname(common_dir)
    if env.get('PYTHONPATH'):
        env['PYTHONPATH'] = os.pathsep.join([root_dir, env['PYTHONPATH']])
    else:
        env['PYTHONPATH'] = root_dir
    
    # Run the simulation
    cmd = [python_exe, test_single, param_file]
    
    print("[Sim %d] Starting..." % sim_number)
    start_time = time.time()
    
    try:
        if show_output:
            # Run simulation with real-time output
            process = subprocess.Popen(
                cmd, 
                env=env, 
                cwd=common_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Stream output in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    # Add simulation number prefix but preserve original formatting
                    # Check if line already has special formatting like [RandomBurst] or [DynamicMatrix]
                    if line.startswith('[') and ']' in line:
                        # Insert sim number after the opening bracket
                        bracket_end = line.index(']')
                        formatted_line = "[Sim %d]%s" % (sim_number, line)
                    else:
                        # Regular line - just add prefix
                        formatted_line = "[Sim %d] %s" % (sim_number, line)
                    print(formatted_line)
                    output_lines.append(line)
            
            process.wait()
            
        else:
            # Run simulation quietly (original behavior)
            process = subprocess.Popen(
                cmd, 
                env=env, 
                cwd=common_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Collect output
            output, _ = process.communicate()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print("[Sim %d] Process completed successfully" % sim_number)
            # Don't check for result file - just trust that the simulation worked
            # The files are being saved in backup/test_single, not in numbered directories
            return (sim_number, True, "Success", elapsed)
        else:
            print("[Sim %d] Failed with code %d" % 
                  (sim_number, process.returncode))
            if not show_output:
                print("Last output:", output[-500:] if len(output) > 500 else output)
            return (sim_number, False, "Return code: %d" % process.returncode, elapsed)
            
    except Exception as e:
        elapsed = time.time() - start_time
        print("[Sim %d] Error: %s" % (sim_number, str(e)))
        return (sim_number, False, str(e), elapsed)

def run_parallel_simulations(num_sims=8, num_processes=None, show_output=True, organize_results=True):
    """
    Run multiple SORN simulations in parallel
    
    Parameters:
        num_sims: Total number of simulations to run
        num_processes: Number of parallel processes (None = use CPU count)
        show_output: Show real-time output from simulations
        organize_results: Move results to batch folder
    """
    # Get paths
    common_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(common_dir)
    
    print("Common directory:", common_dir)
    print("Root directory:", root_dir)
    
    # Find Python 2.7 and verify files
    python_exe = find_python27()
    print("Using Python:", python_exe)
    
    test_single = os.path.join(common_dir, "test_single.py")
    param_file = os.path.join(root_dir, "delpapa", "param_FrozenPlasticity.py")
    
    if not os.path.exists(test_single):
        raise Exception("Cannot find test_single.py at %s" % test_single)
    if not os.path.exists(param_file):
        raise Exception("Cannot find param_FrozenPlasticity.py at %s" % param_file)
    
    print("Test script:", test_single)
    print("Parameter file:", param_file)
    
    # Get batch folder name from parameters
    batch_folder = get_batch_folder_name(param_file)
    print("Batch folder will be:", batch_folder)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), num_sims)
    else:
        num_processes = min(num_processes, num_sims)
    
    print("\nRunning %d simulations using %d parallel processes" % (num_sims, num_processes))
    if show_output:
        print("Real-time output enabled (use --quiet to disable)")
    else:
        print("Running in quiet mode")
    print("-" * 60)
    
    # Create arguments for each simulation
    simulation_args = []
    for i in range(num_sims):
        sim_number = i + 1
        args = (sim_number, python_exe, test_single, param_file, common_dir, show_output)
        simulation_args.append(args)
    
    # Create process pool and run simulations
    start_time = time.time()
    
    # Use Pool for parallel execution
    pool = multiprocessing.Pool(processes=num_processes)
    
    try:
        # Run simulations and collect results
        results = pool.map(run_single_simulation, simulation_args)
        
    finally:
        # Clean up
        pool.close()
        pool.join()
    
    # Summarize results
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _, _ in results if success)
    failed = num_sims - successful
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print("Total simulations: %d" % num_sims)
    print("Successful: %d" % successful)
    print("Failed: %d" % failed)
    print("Total time: %.1f seconds" % total_time)
    print("Average time per simulation: %.1f seconds" % (total_time / num_sims))
    print("Speedup vs sequential: %.1fx" % (sum(t for _, _, _, t in results) / total_time))
    
    # Print failed simulations
    if failed > 0:
        print("\nFailed simulations:")
        for sim_num, success, error, _ in results:
            if not success:
                print("  Simulation %d: %s" % (sim_num, error))
    
    # Organize results into batch folder
    if organize_results:
        print("\n" + "="*60)
        print("ORGANIZING RESULTS")
        print("="*60)
        batch_path = organize_simulation_results(batch_folder, num_sims, root_dir)
        if batch_path:
            print("Results organized in: %s" % batch_path)
        else:
            print("Failed to organize results")
    
    return results

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Run SORN simulations in parallel')
    parser.add_argument('-n', '--num-sims', type=int, default=15,
                        help='Number of simulations to run (default: 15)')
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Number of parallel processes (default: auto)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Run in quiet mode (no simulation output)')
    parser.add_argument('--no-organize', action='store_true',
                        help='Do not organize results into batch folder')
    
    args = parser.parse_args()
    
    # Change to common directory
    common_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(common_dir)
    
    # Run simulations
    results = run_parallel_simulations(args.num_sims, args.processes, 
                                     show_output=not args.quiet,
                                     organize_results=not args.no_organize)
    
    # Return exit code based on results
    failed = sum(1 for _, success, _, _ in results if not success)
    sys.exit(failed)

if __name__ == "__main__":
    main()
