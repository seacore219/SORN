import os
import sys
import subprocess
import shutil
import time

def run_multiple_simulations(num_sims=10):
    """Run test_single.py multiple times for N=200"""
    # Get absolute paths
    common_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(common_dir)
    
    # Clean up existing directories
    for i in range(1, num_sims + 1):
        sim_dir = os.path.join(root_dir, str(i))
        if os.path.exists(sim_dir):
            shutil.rmtree(sim_dir)
        os.makedirs(os.path.join(sim_dir, 'common'))
    
    # Set up Python path to include project root and delpapa
    env = dict(os.environ)
    if env.get('PYTHONPATH'):
        env['PYTHONPATH'] = os.pathsep.join([root_dir, env['PYTHONPATH']])
    else:
        env['PYTHONPATH'] = root_dir
        
    # Change to common directory
    os.chdir(common_dir)
    
    for i in range(num_sims):
        sim_number = i + 1
        print "\nStarting simulation %d of %d" % (sim_number, num_sims)
        
        # Use relative paths since we're in the common directory
        cmd = [
            "C:\\Python27\\python.exe",
            "test_single.py",
            "../delpapa/param_Zheng2013.py"
        ]
        print "Running:", " ".join(cmd)
        print "PYTHONPATH =", env['PYTHONPATH']
        
        result = subprocess.call(cmd, env=env)
        
        if result == 0:
            result_file = os.path.join(root_dir, str(sim_number), 'common', 'result.h5')
            if os.path.exists(result_file):
                print "Successfully created", result_file
            else:
                print "Failed to create result file"
        else:
            print "Simulation failed with code", result
        
        # Brief pause between simulations
        time.sleep(1)

if __name__ == "__main__":
    run_multiple_simulations(num_sims=50)