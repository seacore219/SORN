import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_with_linear_fit(x_values, y_values, output_filename='widom_line_perc.pdf'):
    """
    Plot β versus α values with a linear fit line and save as PDF.
    No LaTeX required version.
    
    Parameters:
    -----------
    x_values : array-like
        Array of β coordinates
    y_values : array-like
        Array of α values
    output_filename : str, optional
        Filename for the output PDF (default: 'widom_line_perc.pdf')
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.array(x_values)
    y = np.array(y_values)
    
    # Create the figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot the raw data points
    plt.scatter(x, y, color='blue', alpha=0.7, label='Data Points')
    
    # Calculate the linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Create the line of best fit
    line_x = np.linspace(min(x), max(x), 100)
    line_y = slope * line_x + intercept
    
    # Plot the line of best fit
    plt.plot(line_x, line_y, 'r-', label=f'Fit: slope = {slope:.4f} intercept = {intercept:.4f}')
    
    # Add equation and R-squared value to the plot
    plt.annotate(f'R² = {r_value**2:.4f}', 
                 xy=(0.05, 0.9), 
                 xycoords='axes fraction', 
                 fontsize=12)
    
    # Add labels and legend - using Unicode instead of LaTeX
    plt.xlabel('β (τ̃ₛ)')  # Beta with tau tilde subscript S in parentheses
    plt.ylabel('α (τ̃ₜ)')  # Alpha with tau tilde subscript T in parentheses
    plt.title('Widom Line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f'Plot saved as {output_filename}')
    
    # Show the plot (optional, comment out if not needed)
    plt.show()
    
    # Return the fit parameters
    return {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}

# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual data
    #x_data = np.array([1.47, 1.39, 1.34])
    #y_data = np.array([1.62, 1.53, 1.46])

    x_data = np.array([1.477, 1.402, 1.239])
    y_data = np.array([1.615, 1.534, 1.329])
    
    # Plot with linear fit and save as PDF
    fit_results = plot_with_linear_fit(x_data, y_data, 'widom_line_perc.pdf')
    
    print("Widom Line")
    print(f"Slope: {fit_results['slope']:.4f}")
    print(f"Intercept: {fit_results['intercept']:.4f}")
    print(f"R-squared: {fit_results['r_squared']:.4f}")
