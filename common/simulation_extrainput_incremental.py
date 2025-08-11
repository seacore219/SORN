#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel ExtraInputNew Simulation Runner with Incremental Saving
================================================================

Usage: python simulation_extrainput_incremental.py -n 10 -p 10
"""

from __future__ import division
import os
import sys
import time
import multiprocessing
import subprocess
import argparse
import shutil
from datetime import datetime

def find_python27():
    """Find Python 2.7 executable"""
    candidates = ['python2.7', 'python2', 'python']
    for candidate in candidates:
        try:
            result = subprocess.check_output([candidate, '--version'], 
                                           stderr=subprocess.STDOUT, 
                                           universal_newlines=True)
            if '2.7' in result:
                return candidate
        except (subprocess.CalledProcessError, OSError):
            continue
    
    raise Exception("Could not find Python 2.7")

def get_batch_folder_name(param_file_path):
    """Generate batch folder name based on parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "ExtraInputNew_batch_%s" % timestamp

def organize_simulation_results(batch_folder, num_sims, root_dir):
    """Organize simulation results into batch folder"""
    batch_path = os.path.join(root_dir, "results", batch_folder)
    backup_path = os.path.join(root_dir, "backup")
    
    if not os.path.exists(backup_path):
        print("No backup directory found")
        return None
    
    # Create batch directory
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    
    moved_count = 0
    
    # Find and move simulation directories
    for item in os.listdir(backup_path):
        item_path = os.path.join(backup_path, item)
        if os.path.isdir(item_path) and "test_single_extrainput_incremental" in item:
            dest_path = os.path.join(batch_path, "sim_%03d_%s" % (moved_count + 1, item))
            try:
                shutil.move(item_path, dest_path)
                moved_count += 1
                print("  Moved simulation %d to %s" % (moved_count, dest_path))
            except Exception as e:
                print("  Error moving %s: %s" % (item, str(e)))
    
    print("Moved %d simulation folders to: %s" % (moved_count, batch_path))
    return batch_path

def run_single_simulation(args):
    """Run a single ExtraInputNew simulation with incremental saving"""
    sim_number, python_exe, test_single, param_file, common_dir, show_output = args

    # Stagger start times
    stagger_delay = (sim_number - 1) * 2.0  # 2 seconds per simulation
    if stagger_delay > 0:
        print("[Sim %d] Waiting %.1f seconds before starting..." % (sim_number, stagger_delay))
        time.sleep(stagger_delay)
    
    # Set up environment
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
    
    print("[Sim %d] Starting ExtraInputNew simulation..." % sim_number)
    start_time = time.time()
    
    try:
        if show_output:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd, 
                env=env, 
                cwd=common_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip()
                    formatted_line = "[Sim %d] %s" % (sim_number, line)
                    print(formatted_line)
                    output_lines.append(line)
            
            process.wait()
            
        else:
            # Run quietly
            process = subprocess.Popen(
                cmd, 
                env=env, 
                cwd=common_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            output, _ = process.communicate()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print("[Sim %d] ExtraInputNew simulation completed successfully in %.1f seconds" % 
                  (sim_number, elapsed))
            return (sim_number, True, "Success", elapsed)
        else:
            print("[Sim %d] Failed with return code %d" % 
                  (sim_number, process.returncode))
            if not show_output:
                print("Last output:", output[-500:] if len(output) > 500 else output)
            return (sim_number, False, "Return code: %d" % process.returncode, elapsed)
            
    except Exception as e:
        elapsed = time.time() - start_time
        print("[Sim %d] Error: %s" % (sim_number, str(e)))
        return (sim_number, False, str(e), elapsed)

def run_parallel_simulations(num_sims=10, num_processes=None, show_output=True, 
                           organize_results=True):
    """Run multiple ExtraInputNew simulations in parallel"""
    
    # Get paths
    common_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(common_dir)
    
    print("Common directory:", common_dir)
    print("Root directory:", root_dir)
    
    # Find Python 2.7
    python_exe = find_python27()
    print("Using Python:", python_exe)
    
    # Use our incremental test script
    test_single = os.path.join(common_dir, "test_single_extrainput_incremental.py")
    param_file = os.path.join(root_dir, "delpapa", "param_ExtraInputNew.py")
    
    if not os.path.exists(test_single):
        raise Exception("Cannot find test_single_extrainput_incremental.py at %s" % test_single)
    if not os.path.exists(param_file):
        raise Exception("Cannot find param_ExtraInputNew.py at %s" % param_file)
    
    print("Test script:", test_single)
    print("Parameter file:", param_file)
    print("Experiment: ExtraInputNew with Incremental Saving")
    
    # Get batch folder name
    batch_folder = get_batch_folder_name(param_file)
    print("Batch folder will be:", batch_folder)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), num_sims)
    else:
        num_processes = min(num_processes, num_sims)
    
    print("\nRunning %d ExtraInputNew simulations using %d parallel processes" % 
          (num_sims, num_processes))
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
    print("EXTRAINPUT INCREMENTAL SIMULATION SUMMARY")
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
            print("\nNext steps:")
            print("1. Download results to local machine")
            print("2. Run concatenate_chunks.py to combine spike chunks")
            print("3. Run cleanup_remote_chunks.py to delete old chunks on RIS")
        else:
            print("Failed to organize results")
    
    return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run ExtraInputNew simulations with incremental saving')
    parser.add_argument('-n', '--num-sims', type=int, default=10,
                        help='Number of simulations to run (default: 10)')
    parser.add_argument('-p', '--processes', type=int, default=10,
                        help='Number of parallel processes (default: 10)')
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