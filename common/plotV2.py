import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import re
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set up the style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class NeuralNetworkParameterAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data = None
        self.output_dir = Path(base_path)  # Save outputs in the base directory
        self.metrics = [
            'Overall_Susceptibility', 'Overall_Rho', 'Overall_CV',
            'Branching_Ratio_Method_1', 'Branching_Ratio_Method_2',
            'Branching_Ratio_Priesman', 'Pearson_Kappa',
            'D2_Correlation_Dimension', 'D2_AR_Order'
        ]
        self.metric_names = {
            'Overall_Susceptibility': 'Susceptibility',
            'Overall_Rho': 'Activity (ρ)',
            'Overall_CV': 'CV',
            'Branching_Ratio_Method_1': 'Branch Param',
            'Branching_Ratio_Method_2': 'Naive BR',
            'Branching_Ratio_Priesman': 'Priesman BR',
            'Pearson_Kappa': 'Pearson κ',
            'D2_Correlation_Dimension': 'D2 Correlation',
            'D2_AR_Order': 'D2 Order'
        }
        
    def extract_parameters(self, folder_name):
        """Extract h_ip, fp, cde, and cdi from folder name"""
        pattern = r'batch_hip([\d.]+)_fp([\d.]+)_cde([\d.]+)_cdi([\d.]+)'
        match = re.match(pattern, folder_name)
        if match:
            return {
                'h_ip': float(match.group(1)),
                'fp': float(match.group(2)),
                'cde': float(match.group(3)),
                'cdi': float(match.group(4))
            }
        return None
    
    def load_all_data(self):
        """Load all Overall_Stats.csv files from subdirectories"""
        all_data = []
        available_metrics = set()
        
        # Iterate through all batch folders
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith('batch_'):
                # Extract parameters from folder name
                params = self.extract_parameters(folder.name)
                if params:
                    # Look for Overall_Stats.csv
                    csv_path = folder / 'Overall_Stats.csv'
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path)
                            # Track which metrics are available
                            available_metrics.update(df.columns)
                            # Add parameter columns
                            for key, value in params.items():
                                df[key] = value
                            all_data.append(df)
                            print(f"Loaded data from {folder.name}")
                        except Exception as e:
                            print(f"Error loading {csv_path}: {e}")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            
            # Update metrics list to only include available columns
            self.metrics = [m for m in self.metrics if m in self.data.columns]
            
            # Convert metric columns to numeric, replacing any non-numeric values with NaN
            for metric in self.metrics:
                if metric in self.data.columns:
                    self.data[metric] = pd.to_numeric(self.data[metric], errors='coerce')
                    # Fill NaN values with 0
                    self.data[metric].fillna(0, inplace=True)
            
            # Ensure parameter columns are numeric
            for param in ['h_ip', 'fp', 'cde', 'cdi']:
                if param in self.data.columns:
                    self.data[param] = pd.to_numeric(self.data[param], errors='coerce')
            
            print(f"\nTotal records loaded: {len(self.data)}")
            print(f"Parameters found: h_ip, fp, cde, cdi")
            print(f"Available metrics: {', '.join([self.metric_names[m] for m in self.metrics])}")
            
            # Report missing metrics
            missing_metrics = [self.metric_names[m] for m in self.metric_names.keys() 
                             if m not in self.metrics]
            if missing_metrics:
                print(f"Missing metrics (will be skipped): {', '.join(missing_metrics)}")
        else:
            print("No data files found!")

    def load_individual_stats_and_calculate_errors(self):
        """
        Load all Individual_Stats CSV files and calculate means and standard errors
        
        Returns:
            tuple: (mu_values, means_dict, errors_dict) or (None, None, None) if no files found
        """
        
        # Find all Individual_Stats CSV files
        csv_pattern = str(self.base_path / 'Individual_Stats_Mu_*.csv')
        csv_files = glob.glob(csv_pattern)
        
        # Also check subdirectories for Individual_Stats files
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith('batch_'):
                csv_pattern_sub = str(folder / 'Individual_Stats_Mu_*.csv')
                csv_files.extend(glob.glob(csv_pattern_sub))
        
        if not csv_files:
            print(f"No Individual_Stats CSV files found in {self.base_path}")
            return None, None, None
        
        print(f"Found {len(csv_files)} Individual_Stats CSV files")
        
        # Lists to store aggregated results
        mu_values = []
        means = {}
        std_errors = {}
        
        # Initialize dictionaries for each metric
        metrics = ['Susceptibility', 'Rho', 'CV', 'Branching_Ratio_Method_1', 
                   'Branching_Ratio_Method_2', 'Branching_Ratio_Priesman', 
                   'Pearson_Kappa', 'D2_Correlation_Dimension', 'D2_AR_Order']
        
        for metric in metrics:
            means[metric] = []
            std_errors[metric] = []
        
        # Process each individual stats file
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                
                # Extract mu value from filename
                filename = os.path.basename(csv_file)
                mu_match = filename.replace('Individual_Stats_Mu_', '').replace('.csv', '')
                mu_val = float(mu_match)
                mu_values.append(mu_val)
                
                print(f"Processing {filename} (mu={mu_val}) with {len(df)} individual runs")
                
                # Calculate mean and standard error for each metric
                for metric in metrics:
                    if metric in df.columns:
                        individual_values = df[metric].dropna()  # Remove NaN values
                        
                        if len(individual_values) > 0:
                            mean_val = np.mean(individual_values)
                            if len(individual_values) > 1:
                                std_error = np.std(individual_values, ddof=1) / np.sqrt(len(individual_values))
                            else:
                                std_error = 0
                        else:
                            mean_val = np.nan
                            std_error = 0
                        
                        means[metric].append(mean_val)
                        std_errors[metric].append(std_error)
                        
                        if not np.isnan(mean_val):
                            print(f"  {metric}: mean={mean_val:.4f}, std_error={std_error:.4f}, n={len(individual_values)}")
                    else:
                        means[metric].append(np.nan)
                        std_errors[metric].append(0)
                        print(f"  {metric}: NOT FOUND in data")
                        
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return mu_values, means, std_errors

    def create_simple_error_bar_plots(self, save_path=None):
        """Create simple plots with error bars from Individual_Stats files"""
        if save_path is None:
            save_path = self.output_dir / 'simple_error_bar_plots.pdf'
        
        # Load individual stats and calculate errors
        print("\nLoading individual statistics files for error bar analysis...")
        mu_values, means_dict, errors_dict = self.load_individual_stats_and_calculate_errors()
        
        if mu_values is None:
            print("No Individual_Stats files found. Cannot create error bar plots.")
            return
        
        # Convert to arrays for plotting
        mu_values = np.array(mu_values)
        
        # Create the plotting data with error bars
        plotting_data_with_errors = [
            (means_dict['Susceptibility'], errors_dict['Susceptibility'], 'Susceptibility ($\\chi$)'),
            (means_dict['Rho'], errors_dict['Rho'], 'Rho ($\\rho$)'),
            (means_dict['CV'], errors_dict['CV'], 'CV'),
            (means_dict['Branching_Ratio_Method_1'], errors_dict['Branching_Ratio_Method_1'], 'Branching Ratio (Method 1)'),
            (means_dict['Branching_Ratio_Method_2'], errors_dict['Branching_Ratio_Method_2'], 'Branching Ratio (Method 2)'),
            (means_dict['Branching_Ratio_Priesman'], errors_dict['Branching_Ratio_Priesman'], 'Branching Ratio (Priesman)'),
            (means_dict['Pearson_Kappa'], errors_dict['Pearson_Kappa'], 'Pearson $\\kappa$'),
            (means_dict['D2_Correlation_Dimension'], errors_dict['D2_Correlation_Dimension'], 'D2 Correlation Dimension'),
            (means_dict['D2_AR_Order'], errors_dict['D2_AR_Order'], 'D2 AR Order')
        ]
        
        with PdfPages(save_path) as pdf:
            # Create figure parameters
            width = 6
            height = width * 1.5
            
            # Create a figure with a grid of subplots (5 rows, 2 columns)
            fig, axes = plt.subplots(5, 2, figsize=(width * 2, height * 5))
            fig.suptitle('Neural Network Metrics with Error Bars from Individual Runs', fontsize=16, y=0.995)
            axes = axes.flatten()
            
            # Plot each dataset with error bars
            for idx, (data, errors, title) in enumerate(plotting_data_with_errors):
                if idx < len(axes):
                    ax = axes[idx]
                    
                    # Convert to NumPy arrays for filtering
                    mu_clean = np.array(mu_values)
                    data_clean = np.array(data)
                    errors_clean = np.array(errors)
                    
                    # Remove NaN or None values
                    valid_mask = ~np.isnan(data_clean) & ~np.isnan(mu_clean)
                    mu_clean = mu_clean[valid_mask]
                    data_clean = data_clean[valid_mask]
                    errors_clean = errors_clean[valid_mask]
                    
                    if len(mu_clean) > 0:  # Only plot if we have valid data
                        # Plot with error bars
                        ax.errorbar(mu_clean, data_clean, yerr=errors_clean, 
                                   marker='o', linestyle='-', markersize=6, linewidth=2,
                                   capsize=4, capthick=1.5, elinewidth=1.5, 
                                   color='darkblue', ecolor='lightblue', alpha=0.8)
                        
                        ax.set_title(title, fontsize=11, pad=10, weight='bold')
                        ax.set_xlabel(r'$\mu$', fontsize=10, weight='semibold')
                        ax.set_ylabel('Value', fontsize=10, weight='semibold')
                        
                        # Customize grid and spines
                        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
                        for spine in ax.spines.values():
                            spine.set_linewidth(1.2)
                            spine.set_color('darkgray')
                        
                        # Set tick parameters
                        ax.tick_params(axis='both', which='major', labelsize=9, 
                                      width=1.2, length=5, color='darkgray')
                        
                        # Add some padding to the plot
                        if len(mu_clean) > 1:
                            x_range = max(mu_clean) - min(mu_clean)
                            ax.set_xlim(min(mu_clean) - 0.05*x_range, max(mu_clean) + 0.05*x_range)
                        
                        if len(data_clean) > 1:
                            y_range = max(data_clean) - min(data_clean)
                            y_center = np.mean(data_clean)
                            if y_range > 0:
                                ax.set_ylim(y_center - 0.6*y_range, y_center + 0.6*y_range)
                        
                    else:
                        # If no valid data, show a message
                        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12, color='red')
                        ax.set_title(title, fontsize=11, pad=10, weight='bold')
                        ax.set_xlabel(r'$\mu$', fontsize=10)
                        ax.set_ylabel('Value', fontsize=10)
            
            # Remove any unused subplot slots
            for idx in range(len(plotting_data_with_errors), len(axes)):
                fig.delaxes(axes[idx])
            
            # Adjust spacing between subplots
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"Error bar plots saved to {save_path}")
        
        # Also create summary table
        summary_output = self.output_dir / 'statistical_summary_with_errors.txt'
        with open(summary_output, 'w') as f:
            f.write("STATISTICAL SUMMARY FROM INDIVIDUAL SIMULATION RUNS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis performed on {len(mu_values)} different μ values\n")
            f.write(f"μ range: {min(mu_values):.3f} to {max(mu_values):.3f}\n\n")
            
            metrics_ordered = [
                ('Susceptibility', 'Susceptibility ($\\chi$)'),
                ('Rho', 'Rho ($\\rho$)'),
                ('CV', 'CV'),
                ('Branching_Ratio_Method_1', 'Branching Ratio (Method 1)'),
                ('Branching_Ratio_Method_2', 'Branching Ratio (Method 2)'),
                ('Branching_Ratio_Priesman', 'Branching Ratio (Priesman)'),
                ('Pearson_Kappa', 'Pearson κ'),
                ('D2_Correlation_Dimension', 'D2 Correlation Dimension'),
                ('D2_AR_Order', 'D2 AR Order')
            ]
            
            for metric_key, metric_name in metrics_ordered:
                f.write(f"{metric_name}:\n")
                f.write("-" * 50 + "\n")
                
                mu_clean = np.array(mu_values)
                data_clean = np.array(means_dict[metric_key])
                errors_clean = np.array(errors_dict[metric_key])
                
                valid_mask = ~np.isnan(data_clean) & ~np.isnan(mu_clean)
                mu_clean = mu_clean[valid_mask]
                data_clean = data_clean[valid_mask]
                errors_clean = errors_clean[valid_mask]
                
                if len(mu_clean) > 0:
                    for i, (mu, mean_val, std_err) in enumerate(zip(mu_clean, data_clean, errors_clean)):
                        f.write(f"  μ = {mu:.3f}: {mean_val:.6f} ± {std_err:.6f}\n")
                    
                    # Add summary statistics
                    f.write(f"\n  Overall range: {np.min(data_clean):.6f} to {np.max(data_clean):.6f}\n")
                    f.write(f"  Mean across all μ: {np.mean(data_clean):.6f}\n")
                    f.write(f"  Std across all μ: {np.std(data_clean):.6f}\n")
                else:
                    f.write("  No valid data available\n")
                f.write("\n")
        
        print(f"Statistical summary saved to {summary_output}")
            
    def create_parameter_comparison_matrix(self, save_path=None):
        """Create plots showing how metrics change with h_ip for different parameter settings"""
        if save_path is None:
            save_path = self.output_dir / 'parameter_comparison_matrix.pdf'
            
        if self.data is None:
            print("No data loaded!")
            return
            
        with PdfPages(save_path) as pdf:
            # Create a figure for each metric
            for metric in self.metrics:
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                fig.suptitle(f'{self.metric_names[metric]} vs h_ip for Different Parameter Settings', 
                           fontsize=16, y=0.995)
                
                # 1. Effect of fp (top-left)
                ax = axes[0, 0]
                # Group by fp values
                unique_fps = sorted(self.data['fp'].unique())
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_fps)))
                
                for i, fp_val in enumerate(unique_fps):
                    # Get baseline data (cde=0, cdi=0)
                    data_subset = self.data[
                        (self.data['fp'] == fp_val) & 
                        (self.data['cde'] == 0) & 
                        (self.data['cdi'] == 0)
                    ].sort_values('h_ip')
                    
                    if not data_subset.empty:
                        ax.plot(data_subset['h_ip'], data_subset[metric], 
                               'o-', color=colors[i], label=f'fp={fp_val}',
                               markersize=8, linewidth=2, alpha=0.8)
                
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(self.metric_names[metric], fontsize=12)
                ax.set_title('Effect of fp (cde=0, cdi=0)', fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # 2. Effect of cde (top-right)
                ax = axes[0, 1]
                unique_cdes = sorted(self.data['cde'].unique())
                colors = plt.cm.plasma(np.linspace(0, 1, len(unique_cdes)))
                
                for i, cde_val in enumerate(unique_cdes):
                    # Get baseline data (fp=0, cdi=0)
                    data_subset = self.data[
                        (self.data['fp'] == 0) & 
                        (self.data['cde'] == cde_val) & 
                        (self.data['cdi'] == 0)
                    ].sort_values('h_ip')
                    
                    if not data_subset.empty:
                        ax.plot(data_subset['h_ip'], data_subset[metric], 
                               's-', color=colors[i], label=f'cde={cde_val}',
                               markersize=8, linewidth=2, alpha=0.8)
                
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(self.metric_names[metric], fontsize=12)
                ax.set_title('Effect of cde (fp=0, cdi=0)', fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # 3. Effect of cdi (bottom-left)
                ax = axes[1, 0]
                unique_cdis = sorted(self.data['cdi'].unique())
                colors = plt.cm.cool(np.linspace(0, 1, len(unique_cdis)))
                
                for i, cdi_val in enumerate(unique_cdis):
                    # Get baseline data (fp=0, cde=0)
                    data_subset = self.data[
                        (self.data['fp'] == 0) & 
                        (self.data['cde'] == 0) & 
                        (self.data['cdi'] == cdi_val)
                    ].sort_values('h_ip')
                    
                    if not data_subset.empty:
                        ax.plot(data_subset['h_ip'], data_subset[metric], 
                               '^-', color=colors[i], label=f'cdi={cdi_val}',
                               markersize=8, linewidth=2, alpha=0.8)
                
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(self.metric_names[metric], fontsize=12)
                ax.set_title('Effect of cdi (fp=0, cde=0)', fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # 4. Combined effects (bottom-right)
                ax = axes[1, 1]
                
                # Plot baseline
                baseline = self.data[
                    (self.data['fp'] == 0) & 
                    (self.data['cde'] == 0) & 
                    (self.data['cdi'] == 0)
                ].sort_values('h_ip')
                if not baseline.empty:
                    ax.plot(baseline['h_ip'], baseline[metric], 
                           'ko-', label='Baseline', markersize=8, linewidth=3)
                
                # Plot some interesting combinations
                interesting_combos = [
                    {'fp': 0.01, 'cde': 0.01, 'cdi': 0.01, 'color': 'red', 'marker': 'D'},
                    {'fp': 0.05, 'cde': 0.01, 'cdi': 0, 'color': 'blue', 'marker': 'v'},
                    {'fp': 0.1, 'cde': 0.1, 'cdi': 0, 'color': 'green', 'marker': 'P'},
                ]
                
                for combo in interesting_combos:
                    data_subset = self.data[
                        (self.data['fp'] == combo['fp']) & 
                        (self.data['cde'] == combo['cde']) & 
                        (self.data['cdi'] == combo['cdi'])
                    ].sort_values('h_ip')
                    
                    if not data_subset.empty:
                        label = f"fp={combo['fp']}, cde={combo['cde']}, cdi={combo['cdi']}"
                        ax.plot(data_subset['h_ip'], data_subset[metric], 
                               f"{combo['marker']}-", color=combo['color'], 
                               label=label, markersize=8, linewidth=2, alpha=0.8)
                
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(self.metric_names[metric], fontsize=12)
                ax.set_title('Combined Effects', fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Optimize subplot spacing for better page composition
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
                
        print(f"Parameter comparison matrix saved to {save_path}")
    
    def create_heatmap_summary(self, save_path=None):
        """Create heatmaps showing average metric values for different parameter combinations"""
        if save_path is None:
            save_path = self.output_dir / 'parameter_heatmap_summary.pdf'
            
        if self.data is None:
            print("No data loaded!")
            return
            
        with PdfPages(save_path) as pdf:
            # Create 2x2 heatmaps for different parameter pairs
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Average Metric Values for Parameter Combinations', fontsize=16)
            
            param_pairs = [
                ('h_ip', 'fp'),
                ('h_ip', 'cde'),
                ('fp', 'cde'),
                ('cde', 'cdi')
            ]
            
            for idx, (param1, param2) in enumerate(param_pairs):
                ax = axes[idx // 2, idx % 2]
                
                # Create pivot table for average susceptibility (you can change the metric)
                pivot_data = self.data.pivot_table(
                    values='Overall_Susceptibility',
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(pivot_data, annot=True, fmt='.2e', cmap='YlOrRd',
                           ax=ax, cbar_kws={'label': 'Avg Susceptibility'})
                ax.set_title(f'{param1.upper()} vs {param2.upper()}', fontsize=14)
                ax.set_xlabel(param2.upper(), fontsize=12)
                ax.set_ylabel(param1.upper(), fontsize=12)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Create individual heatmaps for each metric
            for metric in self.metrics:
                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                fig.suptitle(f'{self.metric_names[metric]} - Parameter Combinations', fontsize=16)
                
                for idx, (param1, param2) in enumerate(param_pairs):
                    ax = axes[idx // 2, idx % 2]
                    
                    pivot_data = self.data.pivot_table(
                        values=metric,
                        index=param1,
                        columns=param2,
                        aggfunc='mean'
                    )
                    
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='coolwarm',
                               ax=ax, cbar_kws={'label': self.metric_names[metric]})
                    ax.set_title(f'{param1.upper()} vs {param2.upper()}', fontsize=14)
                    ax.set_xlabel(param2.upper(), fontsize=12)
                    ax.set_ylabel(param1.upper(), fontsize=12)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
        print(f"Heatmap summary saved to {save_path}")
    
    def create_baseline_comparison(self, save_path=None):
        """Create visualizations showing how h_ip trends change with different parameters"""
        if save_path is None:
            save_path = self.output_dir / 'baseline_comparison.pdf'
            
        if self.data is None:
            print("No data loaded!")
            return
            
        with PdfPages(save_path) as pdf:
            # 1. Side-by-side comparison of how each parameter modifies h_ip trends
            n_metrics = len(self.metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
            fig.suptitle('How Parameters Modify h_ip Trends', fontsize=16)
            
            if n_metrics == 1:
                axes_flat = [axes]
            else:
                axes_flat = axes.flatten()
            
            for idx, metric in enumerate(self.metrics):
                ax = axes_flat[idx]
                
                # Plot baseline (all other params = 0)
                baseline = self.data[
                    (self.data['fp'] == 0) & 
                    (self.data['cde'] == 0) & 
                    (self.data['cdi'] == 0)
                ].sort_values('h_ip')
                
                if not baseline.empty:
                    ax.plot(baseline['h_ip'], baseline[metric], 
                           'ko-', label='Baseline', markersize=10, linewidth=3, alpha=1.0)
                
                # Overlay different parameter effects
                # fp effect
                fp_data = self.data[
                    (self.data['fp'] > 0) & 
                    (self.data['cde'] == 0) & 
                    (self.data['cdi'] == 0)
                ]
                if not fp_data.empty:
                    for fp_val in sorted(fp_data['fp'].unique())[:3]:  # Top 3 values
                        subset = fp_data[fp_data['fp'] == fp_val].sort_values('h_ip')
                        ax.plot(subset['h_ip'], subset[metric], 
                               'o--', color='red', alpha=0.5 + 0.2*fp_val, 
                               label=f'fp={fp_val}', markersize=6, linewidth=2)
                
                # cde effect
                cde_data = self.data[
                    (self.data['fp'] == 0) & 
                    (self.data['cde'] > 0) & 
                    (self.data['cdi'] == 0)
                ]
                if not cde_data.empty:
                    for cde_val in sorted(cde_data['cde'].unique())[:3]:
                        subset = cde_data[cde_data['cde'] == cde_val].sort_values('h_ip')
                        ax.plot(subset['h_ip'], subset[metric], 
                               's--', color='blue', alpha=0.5 + 0.2*cde_val,
                               label=f'cde={cde_val}', markersize=6, linewidth=2)
                
                ax.set_xlabel('h_ip', fontsize=10)
                ax.set_ylabel(self.metric_names[metric], fontsize=10)
                ax.set_title(self.metric_names[metric], fontsize=12)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
                
                # Add legend only to first plot
                if idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Hide unused subplots
            for idx in range(n_metrics, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 2. Slope analysis - how parameters change the h_ip slope
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle('h_ip Slope Changes by Parameter Settings', fontsize=16)
            axes_flat = axes.flatten()
            
            for idx, metric in enumerate(self.metrics[:9]):  # Max 9 metrics
                ax = axes_flat[idx]
                
                slope_data = []
                
                # Calculate slopes for different parameter combinations
                param_combos = self.data.groupby(['fp', 'cde', 'cdi']).size().reset_index()[['fp', 'cde', 'cdi']]
                
                for _, combo in param_combos.iterrows():
                    subset = self.data[
                        (self.data['fp'] == combo['fp']) & 
                        (self.data['cde'] == combo['cde']) & 
                        (self.data['cdi'] == combo['cdi'])
                    ].sort_values('h_ip')
                    
                    if len(subset) >= 2:  # Need at least 2 points for slope
                        try:
                            # Fit log-linear relationship
                            log_h_ip = np.log10(subset['h_ip'])
                            slope, intercept = np.polyfit(log_h_ip, subset[metric], 1)
                            
                            # Categorize by which parameters are active
                            if combo['fp'] > 0 and combo['cde'] == 0 and combo['cdi'] == 0:
                                category = 'fp only'
                                color = 'red'
                            elif combo['cde'] > 0 and combo['fp'] == 0 and combo['cdi'] == 0:
                                category = 'cde only'
                                color = 'blue'
                            elif combo['cdi'] > 0 and combo['fp'] == 0 and combo['cde'] == 0:
                                category = 'cdi only'
                                color = 'green'
                            elif combo['fp'] == 0 and combo['cde'] == 0 and combo['cdi'] == 0:
                                category = 'baseline'
                                color = 'black'
                            else:
                                category = 'mixed'
                                color = 'purple'
                            
                            slope_data.append({
                                'slope': slope,
                                'category': category,
                                'color': color,
                                'fp': combo['fp'],
                                'cde': combo['cde'],
                                'cdi': combo['cdi']
                            })
                        except:
                            pass
                
                if slope_data:
                    slope_df = pd.DataFrame(slope_data)
                    
                    # Create box plot by category
                    categories = ['baseline', 'fp only', 'cde only', 'cdi only', 'mixed']
                    box_data = []
                    positions = []
                    colors = []
                    
                    for i, cat in enumerate(categories):
                        cat_data = slope_df[slope_df['category'] == cat]['slope']
                        if not cat_data.empty:
                            box_data.append(cat_data)
                            positions.append(i)
                            colors.append(slope_df[slope_df['category'] == cat]['color'].iloc[0])
                    
                    if box_data:
                        bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                        ax.set_xticks(positions)
                        ax.set_xticklabels([categories[i] for i in positions], rotation=45)
                        ax.set_ylabel('Slope (d(metric)/d(log h_ip))', fontsize=10)
                        ax.set_title(self.metric_names[metric], fontsize=12)
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 3. Parameter interaction effects on h_ip trends
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle('Parameter Interaction Effects on Key Metrics vs h_ip', fontsize=16)
            
            # Select 4 key metrics
            key_metrics = self.metrics[:4]
            
            for idx, metric in enumerate(key_metrics):
                ax = axes[idx // 2, idx % 2]
                
                # Create subplots showing interaction effects
                unique_h_ips = sorted(self.data['h_ip'].unique())
                
                # fp vs cde interaction (cdi = 0)
                interaction_data = self.data[self.data['cdi'] == 0]
                
                # Create pivot table for visualization
                pivot = interaction_data.pivot_table(
                    values=metric,
                    index='h_ip',
                    columns=['fp', 'cde'],
                    aggfunc='mean'
                )
                
                # Plot lines for each combination
                for col in pivot.columns:
                    fp_val, cde_val = col
                    if fp_val == 0 and cde_val == 0:
                        ax.plot(pivot.index, pivot[col], 'k-', linewidth=3, 
                               label='Baseline', marker='o', markersize=8)
                    elif fp_val > 0 and cde_val == 0:
                        ax.plot(pivot.index, pivot[col], 'r--', alpha=0.7,
                               label=f'fp={fp_val}', marker='o', markersize=6)
                    elif fp_val == 0 and cde_val > 0:
                        ax.plot(pivot.index, pivot[col], 'b--', alpha=0.7,
                               label=f'cde={cde_val}', marker='s', markersize=6)
                    elif fp_val > 0 and cde_val > 0:
                        ax.plot(pivot.index, pivot[col], 'g:', alpha=0.7,
                               label=f'fp={fp_val}, cde={cde_val}', marker='^', markersize=6)
                
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(self.metric_names[metric], fontsize=12)
                ax.set_title(f'{self.metric_names[metric]} - Parameter Interactions', fontsize=14)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis to show all discrete h_ip values
                unique_h_ips = sorted(pivot.index)
                ax.set_xticks(unique_h_ips)
                ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Limit legend entries
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 6:
                    ax.legend(handles[:6], labels[:6], fontsize=8, loc='best')
                else:
                    ax.legend(fontsize=8, loc='best')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
    def create_color_coded_hip_plots(self, save_path=None):
        """Create h_ip vs metric plots with color coding for different parameter combinations"""
        if save_path is None:
            save_path = self.output_dir / 'hip_color_coded_analysis.pdf'
            
        if self.data is None:
            print("No data loaded!")
            return
            
        # Create a unique identifier for each parameter combination
        self.data['param_combo'] = (
            'fp=' + self.data['fp'].astype(str) + '_' +
            'cde=' + self.data['cde'].astype(str) + '_' +
            'cdi=' + self.data['cdi'].astype(str)
        )
        
        # Get unique parameter combinations
        unique_combos = self.data['param_combo'].unique()
        n_combos = len(unique_combos)
        
        # Create color map
        # Baseline is black, others get colors from a colormap
        colors = {}
        markers = {}
        marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8']
        
        # Helper function to check if a combination is baseline
        def is_baseline(combo_str):
            parts = combo_str.split('_')
            fp_val = float(parts[0].split('=')[1])
            cde_val = float(parts[1].split('=')[1])
            cdi_val = float(parts[2].split('=')[1])
            return fp_val == 0.0 and cde_val == 0.0 and cdi_val == 0.0
        
        baseline_combo = None
        for i, combo in enumerate(sorted(unique_combos)):
            if is_baseline(combo):
                colors[combo] = 'black'
                markers[combo] = 'o'
                baseline_combo = combo
            else:
                # Use a colormap for other combinations
                colors[combo] = plt.cm.tab20(i % 20)
                markers[combo] = marker_list[i % len(marker_list)]
        
        with PdfPages(save_path) as pdf:
            # Page 1: Individual plots for each metric with enhanced poster-quality styling
            n_metrics = len(self.metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            # Create figure with GridSpec for better control
            fig = plt.figure(figsize=(8.5, 11))  # Standard letter size
            fig.patch.set_facecolor('white')
            
            # Create GridSpec with space for title, legend, and plots
            # Adjust height ratios: title space, legend space, then plot rows
            gs = fig.add_gridspec(n_rows + 2, n_cols, 
                                 height_ratios=[0.08, 0.12] + [1.3]*n_rows,  # Much taller plots
                                 hspace=0.55, wspace=0.15)  # Much more vertical space
            
            # Add title in its own space
            title_ax = fig.add_subplot(gs[0, :])
            title_ax.axis('off')
            title_ax.text(0.5, 0.5, 'Neural Network Metrics vs h_ip - Parameter Effects', 
                         fontsize=16, fontweight='bold', ha='center', va='center',
                         transform=title_ax.transAxes)
            
            # Create a subplot for the legend spanning all columns
            legend_ax = fig.add_subplot(gs[1, :])
            legend_ax.axis('off')  # Hide the axes
            
            # Store all lines and labels for the legend
            all_lines = []
            all_labels = []
            
            # Plot baseline first for legend
            if baseline_combo:
                baseline_line, = legend_ax.plot([], [], 'ko-', label='Baseline', 
                                               markersize=3, linewidth=0.8)
                all_lines.append(baseline_line)
                all_labels.append('Baseline')
            
            # Add other combinations to legend
            for combo in sorted(unique_combos):
                if baseline_combo and combo == baseline_combo:
                    continue
                    
                # Parse the combination for a cleaner label
                parts = combo.split('_')
                fp_val = float(parts[0].split('=')[1])
                cde_val = float(parts[1].split('=')[1])
                cdi_val = float(parts[2].split('=')[1])
                
                # Create label showing only non-zero values
                label_parts = []
                if fp_val > 0:
                    label_parts.append(f'fp={fp_val}')
                if cde_val > 0:
                    label_parts.append(f'cde={cde_val}')
                if cdi_val > 0:
                    label_parts.append(f'cdi={cdi_val}')
                label = ', '.join(label_parts) if label_parts else 'All zero'
                
                line, = legend_ax.plot([], [], marker=markers[combo], linestyle='--',
                                      color=colors[combo], label=label,
                                      markersize=3, linewidth=0.8)
                all_lines.append(line)
                all_labels.append(label)
            
            # Create horizontal legend at the top
            legend = legend_ax.legend(all_lines, all_labels, 
                                     loc='center', ncol=min(len(all_lines), 5),  # More columns to make it more compact
                                     fontsize=7, frameon=True, fancybox=True,
                                     shadow=False, borderpad=0.3, columnspacing=0.8,
                                     handlelength=1.5, handletextpad=0.3)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('#2C2C2C')
            legend.get_frame().set_linewidth(1.5)
            
            # Create subplots for metrics
            axes_flat = []
            for idx in range(n_metrics):
                row = (idx // n_cols) + 2  # +2 to skip title and legend rows
                col = idx % n_cols
                ax = fig.add_subplot(gs[row, col])
                axes_flat.append(ax)
                
                metric = self.metrics[idx]
                
                # Clean white background with subtle grey grid
                ax.set_facecolor('white')
                ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8, color='#D0D0D0')
                ax.set_axisbelow(True)
                
                # Clean dark grey spines
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_color('#2C2C2C')
                
                # Plot baseline with clean styling
                if baseline_combo:
                    baseline_data = self.data[self.data['param_combo'] == baseline_combo].sort_values('h_ip')
                    if not baseline_data.empty:
                        ax.plot(baseline_data['h_ip'], baseline_data[metric],
                               'ko-', markersize=3, linewidth=0.8,
                               alpha=0.9, zorder=10, markeredgewidth=0.5, markeredgecolor='white')
                
                # Plot all other combinations with enhanced styling
                for combo in sorted(unique_combos):
                    if baseline_combo and combo == baseline_combo:
                        continue
                        
                    combo_data = self.data[self.data['param_combo'] == combo].sort_values('h_ip')
                    if not combo_data.empty:
                        ax.plot(combo_data['h_ip'], combo_data[metric],
                               marker=markers[combo], linestyle='--',
                               color=colors[combo],
                               markersize=3, linewidth=0.6, alpha=0.85,
                               markeredgewidth=0.4, markeredgecolor='white')
                
                # Clean axis labels and title
                ax.set_xlabel('h_ip', fontsize=9, fontweight='semibold', labelpad=2)  # Closer to axis
                # Remove y-axis label since it's redundant with the title
                ax.set_ylabel('', fontsize=10)
                ax.set_title(self.metric_names[metric], fontsize=11, fontweight='bold', pad=5)
                ax.set_xscale('log')
                
                # Clean tick styling
                ax.tick_params(axis='both', which='major', labelsize=6, width=1.5, length=4)
                ax.tick_params(axis='both', which='minor', width=1, length=2)
                
                # Set x-axis to show exact h_ip values without rounding
                unique_h_ips = sorted(self.data['h_ip'].unique())
                ax.set_xticks(unique_h_ips)
                # Format labels to show exact values
                ax.set_xticklabels([f'{x:.4g}' for x in unique_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=5)
            
            # No need to hide unused subplots since we're creating exact number needed
            
            pdf.savefig(fig, bbox_inches='tight', dpi=150)  # Lower DPI for smaller file size
            plt.close()
            
            
            # Page 2: All metrics overlaid on one plot
            fig, ax = plt.subplots(figsize=(8.5, 11))
            
            # Filter out D2_AR_Order from metrics for this plot
            plot_metrics = [m for m in self.metrics if m != 'D2_AR_Order']
            
            # Define vibrant colors for each metric
            metric_color_map = {
                'Overall_Susceptibility': '#808080',      # Grey
                'Overall_Rho': '#00CED1',                 # Dark Turquoise
                'Overall_CV': '#FFD700',                  # Gold
                'Branching_Ratio_Method_1': '#4B0082',    # Indigo (dark blue-purple)
                'Branching_Ratio_Method_2': '#8B008B',    # Dark Magenta (red-purple)
                'Branching_Ratio_Priesman': '#9932CC',    # Dark Orchid (bright purple)
                'Pearson_Kappa': '#FF4500',               # Orange Red
                'D2_Correlation_Dimension': '#32CD32'     # Lime Green
            }
            
            # Identify branching ratio metrics and D2 correlation for shared scaling
            br_metrics = ['Branching_Ratio_Method_1', 'Branching_Ratio_Method_2', 'Branching_Ratio_Priesman']
            scaled_metrics = br_metrics + ['D2_Correlation_Dimension']  # Add D2 to shared scale
            
            # Find global min/max for branching ratio metrics and D2
            shared_min = float('inf')
            shared_max = float('-inf')
            for metric in scaled_metrics:
                if metric in self.data.columns:
                    shared_min = min(shared_min, self.data[metric].min())
                    shared_max = max(shared_max, self.data[metric].max())
            
            for metric in plot_metrics:
                # Get all data sorted by h_ip
                sorted_data = self.data.sort_values('h_ip')
                
                # Group by h_ip and calculate mean for each h_ip value
                grouped = sorted_data.groupby('h_ip')[metric].mean()
                
                # Normalize based on metric type
                metric_values = grouped.values
                
                if metric in scaled_metrics:
                    # Use same scale for all branching ratio metrics and D2
                    if shared_max != shared_min:
                        normalized_values = (metric_values - shared_min) / (shared_max - shared_min)
                    else:
                        normalized_values = metric_values
                else:
                    # Use individual normalization for other metrics
                    if metric_values.max() != metric_values.min():
                        normalized_values = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())
                    else:
                        normalized_values = metric_values
                
                # Plot the mean line with vibrant color
                ax.plot(grouped.index, normalized_values, 
                       'o-', color=metric_color_map.get(metric, 'gray'), 
                       label=self.metric_names[metric],
                       markersize=8, linewidth=2, alpha=0.9,
                       markeredgecolor='black', markeredgewidth=0.5)
            
            ax.set_xlabel('h_ip', fontsize=12)
            ax.set_ylabel('Normalized Metric Value (0-1)', fontsize=12)
            ax.set_title('All Metrics Overlaid - Co-occurring Changes Across h_ip\n(D2 Correlation and Branching Ratios share the same scale)', 
                        fontsize=14, pad=10)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=9, ncol=3,
                     frameon=True, fancybox=True, shadow=False)
            
            # Set x-axis to show all discrete h_ip values
            unique_h_ips = sorted(self.data['h_ip'].unique())
            ax.set_xticks(unique_h_ips)
            ax.set_xticklabels([f'{x:.3f}' for x in unique_h_ips])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add reference lines
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()
        
        print(f"Color-coded h_ip analysis saved to {save_path}")
    
    def create_correlation_analysis(self, save_path=None):
        """Create correlation matrices between parameters and metrics"""
        if save_path is None:
            save_path = self.output_dir / 'correlation_analysis.pdf'
            
        if self.data is None:
            print("No data loaded!")
            return
            
        # Select relevant columns
        analysis_cols = ['h_ip', 'fp', 'cde', 'cdi'] + self.metrics
        correlation_data = self.data[analysis_cols].corr()
        
        with PdfPages(save_path) as pdf:
            # Full correlation matrix
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Create custom labels
            labels = ['h_ip', 'fp', 'cde', 'cdi'] + [self.metric_names[m] for m in self.metrics]
            
            sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, square=True, linewidths=1,
                       xticklabels=labels, yticklabels=labels,
                       ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Correlation Matrix: Parameters vs Metrics', fontsize=16, pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Parameter-Metric correlation only
            fig, ax = plt.subplots(figsize=(12, 8))
            param_metric_corr = correlation_data.loc[['h_ip', 'fp', 'cde', 'cdi'], self.metrics]
            
            sns.heatmap(param_metric_corr, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, square=True, linewidths=1,
                       xticklabels=[self.metric_names[m] for m in self.metrics],
                       yticklabels=['h_ip', 'fp', 'cde', 'cdi'],
                       ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Parameter-Metric Correlations', fontsize=16, pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
        print(f"Correlation analysis saved to {save_path}")
    
    def create_interactive_dashboard(self):
        """Create an interactive analysis with all visualizations"""
        print("\nStarting Neural Network Parameter Analysis...")
        print("=" * 60)
        
        # Load all data
        self.load_all_data()
        
        if self.data is not None:
            print("\nGenerating visualizations...")
            print("-" * 40)
            
            # Create error bar plots first (NEW FUNCTIONALITY)
            print("Creating error bar plots from individual stats...")
            self.create_simple_error_bar_plots()
            
            # Create other visualizations
            self.create_color_coded_hip_plots()
            
            # Create a summary statistics file
            self.create_summary_statistics()
            
            print("\nAnalysis complete!")
            print("=" * 60)
            print("\nGenerated files:")
            print("- simple_error_bar_plots.pdf (NEW - with error bars)")
            print("- statistical_summary_with_errors.txt (NEW - detailed stats)")
            print("- hip_color_coded_analysis.pdf")
            print("- summary_statistics.txt")
    
    def create_summary_statistics(self, save_path=None):
        """Create a text file with summary statistics"""
        if save_path is None:
            save_path = self.output_dir / 'summary_statistics.txt'
            
        if self.data is None:
            return
            
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("NEURAL NETWORK PARAMETER ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PARAMETER RANGES:\n")
            f.write("-" * 40 + "\n")
            for param in ['h_ip', 'fp', 'cde', 'cdi']:
                f.write(f"{param.upper():>6}: min={self.data[param].min():.4f}, "
                       f"max={self.data[param].max():.4f}, "
                       f"unique values={self.data[param].nunique()}\n")
            
            f.write("\n\nMETRIC STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for metric in self.metrics:
                f.write(f"\n{self.metric_names[metric]}:\n")
                f.write(f"  Mean: {self.data[metric].mean():.6f}\n")
                f.write(f"  Std:  {self.data[metric].std():.6f}\n")
                f.write(f"  Min:  {self.data[metric].min():.6f}\n")
                f.write(f"  Max:  {self.data[metric].max():.6f}\n")
            
            f.write("\n\nTOP CORRELATIONS WITH PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            analysis_cols = ['h_ip', 'fp', 'cde', 'cdi'] + self.metrics
            corr_matrix = self.data[analysis_cols].corr()
            
            for param in ['h_ip', 'fp', 'cde', 'cdi']:
                f.write(f"\n{param.upper()}:\n")
                param_corrs = corr_matrix.loc[param, self.metrics].sort_values(ascending=False)
                for metric, corr in param_corrs.items():
                    f.write(f"  {self.metric_names[metric]:>20}: {corr:>7.3f}\n")
        
        print(f"Summary statistics saved to {save_path}")

# Usage example
if __name__ == "__main__":
    # Set your base path here
    base_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup"
    
    # Create analyzer instance
    analyzer = NeuralNetworkParameterAnalyzer(base_path)
    
    # Run the complete analysis
    analyzer.create_interactive_dashboard()
    
    print("\nTo use this tool with different data:")
    print("1. Update the 'base_path' variable to point to your backup directory")
    print("2. Run the script to generate all visualizations")
    print("3. Check the generated PDF files for comprehensive analysis")
    print("4. NEW: Error bar plots show variability from individual simulation runs")