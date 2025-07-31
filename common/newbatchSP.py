import os
import sys
import subprocess
import shutil
import time

def find_python27_simple():
    """Simpler approach - just try commands in PATH first"""
    for cmd in ["python2.7", "python2", "python27.exe", "python27", "python"]:
        try:
            result = subprocess.check_output([cmd, "--version"], 
                                           stderr=subprocess.STDOUT, 
                                           universal_newlines=True)
            if "Python 2.7" in result:
                return cmd
        except:
            continue
    
    # If not in PATH, try common Windows locations
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

def verify_python():
    return find_python27_simple()

def verify_paths(common_dir, root_dir):
    """Verify all required files and paths exist"""
    test_single = os.path.join(common_dir, "test_single.py")
    param_file = os.path.join(root_dir, "delpapa", "param_FrozenPlasticity.py")
    
    print "Checking test_single.py exists at:", test_single
    print "Checking param file exists at:", param_file
    
    if not os.path.exists(test_single):
        raise Exception("Cannot find test_single.py at %s" % test_single)
    if not os.path.exists(param_file):
        raise Exception("Cannot find param_FrozenPlasticity.py at %s" % param_file)
    return test_single, param_file

def run_simulation(python_exe, test_single, param_file, env, sim_number, total_sims):
    """Run a single simulation with better error handling"""
    try:
        print "\nStarting simulation %d of %d" % (sim_number, total_sims)
        
        cmd = [python_exe, test_single, param_file]
        
        print "Verifying files exist:"
        print "- Python:", os.path.exists(python_exe)
        print "- Test script:", os.path.exists(test_single)
        print "- Param file:", os.path.exists(param_file)
        
        print "Running:", " ".join(cmd)
        print "PYTHONPATH =", env['PYTHONPATH']
        print "Working directory:", os.getcwd()
        
        #process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = subprocess.Popen(cmd, env=env)
        stdout, stderr = process.communicate()
        
        if stdout:
            print "Output:", stdout
        if stderr:
            print "Errors:", stderr
        
        if process.returncode == 0:
            result_file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                                     str(sim_number), 'common', 'result.h5')
            if os.path.exists(result_file):
                print "Successfully created", result_file
                return True
            else:
                print "Failed to create result file"
                return False
        else:
            print "Simulation failed with code", process.returncode
            return False
            
    except Exception as e:
        print "Error in simulation %d: %s" % (sim_number, str(e))
        return False

def run_multiple_simulations(num_sims=25):
    """Run test_single.py multiple times with better error handling"""
    # Get absolute paths
    common_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(common_dir)
    
    print "Common directory:", common_dir
    print "Root directory:", root_dir
    
    try:
        # Verify Python and paths first
        python_exe = verify_python()
        test_single, param_file = verify_paths(common_dir, root_dir)
        
        # Set up Python path
        env = dict(os.environ)
        if env.get('PYTHONPATH'):
            env['PYTHONPATH'] = os.pathsep.join([root_dir, env['PYTHONPATH']])
        else:
            env['PYTHONPATH'] = root_dir
            
        # Change to common directory
        os.chdir(common_dir)
        
        successful_sims = 0
        failed_sims = 0
        
        for i in range(num_sims):
            sim_number = i + 1
            
            if run_simulation(python_exe, test_single, param_file, env, sim_number, num_sims):
                successful_sims += 1
            else:
                failed_sims += 1
            
            # Brief pause between simulations
            time.sleep(1)
            
        print "\nSimulation Summary:"
        print "Successful simulations:", successful_sims
        print "Failed simulations:", failed_sims
            
    except Exception as e:
        print "Error in simulation batch:", str(e)
    finally:
        print "\nSimulation batch completed"

if __name__ == "__main__":
    run_multiple_simulations(num_sims=15)
