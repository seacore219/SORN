import re
import csv

def parse_log_file(input_file, output_file):
    # Pattern to match the INFO lines
    pattern = r'\[INFO\]\s*Step\s*=\s*(\d+),\s*activity_x\s*=\s*(\d+),\s*activity_y\s*=\s*(\d+),\s*Maximum eigenvalue W_EE\s*=\s*([\d.]+),\s*spect norm W_ei\s*=\s*([\d.]+)'
    
    data = []
    
    # Try different encodings
    encodings = ['utf-16', 'utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    
    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                for line in f:
                    # Look for matches in each line
                    match = re.search(pattern, line.strip())
                    if match:
                        # Extract the values
                        step = int(match.group(1))
                        activity_x = int(match.group(2))
                        activity_y = int(match.group(3))
                        eigenvalue = float(match.group(4))
                        spect_norm_W_ei = float(match.group(5))
                        # Add to data list
                        data.append({
                            'Step': step,
                            'activity_x': activity_x,
                            'activity_y': activity_y,
                            'Maximum_eigenvalue_W_EE': eigenvalue,
                            'spect norm W_ei' : spect_norm_W_ei
                        })
                    elif '[INFO]' in line:
                        print(f"Warning: Could not parse line: {line.strip()}")
                
                # If we successfully read the file, break the loop
                break
                
        except UnicodeDecodeError:
            if encoding == encodings[-1]:  # If this was the last encoding to try
                print(f"Could not read the file with any of the attempted encodings: {encodings}")
                return
            continue
    
    # Write to CSV file
    if data:
        fieldnames = ['Step', 'activity_x', 'activity_y', 'Maximum_eigenvalue_W_EE', 'spect norm W_ei']
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully created {output_file} with {len(data)} entries")
    else:
        print("No matching data found in the log file")

# Usage
if __name__ == "__main__":
    input_file = "output2.txt"  # Change this to your input file name
    output_file = "simulation2_data.csv"  # Change this to your desired output file name
    parse_log_file(input_file, output_file)