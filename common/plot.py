import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from pathlib import Path
import re
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
            'D2_Correlation_Dimension', 'D2_AR_Order',
            'AV_Alpha', 'AV_Beta', 'AV_Scaling_Diff'
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
            'D2_AR_Order': 'D2 Order',
            'AV_Alpha': 'AVsize (α)',  
            'AV_Beta': 'AVduration (β)',
            'AV_Scaling_Diff': 'Scaling Diff'
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
        
        # Also store individual stats for error bars
        all_individual_data = []
        
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
                    
                    # NEW: Also look for Individual_Stats_Mu_*.csv files
                    for individual_file in folder.glob('Individual_Stats_Mu_*.csv'):
                        try:
                            ind_df = pd.read_csv(individual_file)
                            # Add parameter columns to individual data too
                            for key, value in params.items():
                                ind_df[key] = value
                            all_individual_data.append(ind_df)
                        except:
                            pass
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            
            # Update metrics list to only include available columns
            self.metrics = [m for m in self.metrics if m in self.data.columns]
            
            # Convert metric columns to numeric, replacing any non-numeric values with NaN
            # Convert metric columns to numeric, replacing any non-numeric values with NaN
            for metric in self.metrics:
                if metric in self.data.columns:
                    self.data[metric] = pd.to_numeric(self.data[metric], errors='coerce')
                    # Don't fill NaN values for AV columns - keep them as NaN
                    if metric not in ['AV_Alpha', 'AV_Beta', 'AV_Scaling_Diff']:
                        # Fill NaN values with 0 only for non-AV metrics
                        self.data[metric].fillna(0, inplace=True)
            
            # Ensure parameter columns are numeric
            for param in ['h_ip', 'fp', 'cde', 'cdi']:
                if param in self.data.columns:
                    self.data[param] = pd.to_numeric(self.data[param], errors='coerce')
            
            # NEW: Process individual data for standard deviations
            if all_individual_data:
                print(f"\nFound {len(all_individual_data)} individual stats files. Computing error bars...")
                combined_individual = pd.concat(all_individual_data, ignore_index=True)
                
                # Map between Overall_Stats names and Individual_Stats names
                metric_mapping = {
                    'Overall_Susceptibility': 'Susceptibility',
                    'Overall_Rho': 'Rho',
                    'Overall_CV': 'CV',
                    'Branching_Ratio_Method_1': 'Branching_Ratio_Method_1',
                    'Branching_Ratio_Method_2': 'Branching_Ratio_Method_2',
                    'Branching_Ratio_Priesman': 'Branching_Ratio_Priesman',
                    'Pearson_Kappa': 'Pearson_Kappa',
                    'D2_Correlation_Dimension': 'D2_Correlation_Dimension'
                }
                
                # Compute standard deviations grouped by parameters
                std_columns = {}
                for overall_metric, individual_metric in metric_mapping.items():
                    if overall_metric in self.metrics and individual_metric in combined_individual.columns:
                        # Group by all parameters and compute std
                        std_data = combined_individual.groupby(['h_ip', 'fp', 'cde', 'cdi'])[individual_metric].std().reset_index()
                        std_data.rename(columns={individual_metric: f'{overall_metric}_std'}, inplace=True)
                        
                        # Merge with main data
                        self.data = pd.merge(self.data, std_data, on=['h_ip', 'fp', 'cde', 'cdi'], how='left')
                        std_columns[overall_metric] = f'{overall_metric}_std'
                
                # Store the std column mapping for later use
                self.std_columns = std_columns
                print(f"Added error bars for {len(std_columns)} metrics")
            else:
                self.std_columns = {}
            
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

    def create_avalanche_scaling_plots(self, save_path=None, fit_h_ip_range=(0.02, 0.08)):
        """
        Create plots for AVsize, AVduration, and difference vs h_ip with fitted lines,
        plus AVsize vs AVduration plot
        
        Parameters:
        -----------
        save_path : Path, optional
            Where to save the PDF file
        fit_h_ip_range : tuple, optional
            (min, max) range of h_ip values to use for fitting the baseline.
            Default is (0.02, 0.08).
            Example: (0.01, 0.09) to fit only points where 0.01 <= h_ip <= 0.09
        """
        if save_path is None:
            save_path = self.output_dir / 'avalanche_scaling_analysis.pdf'
        
        if self.data is None:
            print("No data loaded!")
            return
        
        # Check if AV columns exist
        av_columns = ['AV_Alpha', 'AV_Beta', 'AV_Scaling_Diff']
        available_av_columns = [col for col in av_columns if col in self.data.columns]
        
        if not available_av_columns:
            print(f"No avalanche columns found in data")
            print("Make sure your Overall_Stats.csv files contain AV_Alpha, AV_Beta, and AV_Scaling_Diff columns")
            return
        
        # Create parameter combination identifier
        self.data['param_combo'] = (
            'fp=' + self.data['fp'].astype(str) + '_' +
            'cde=' + self.data['cde'].astype(str) + '_' +
            'cdi=' + self.data['cdi'].astype(str)
        )
        
        # Filter data to only rows that have h_ip and at least one non-zero AV metric
        mask = self.data['h_ip'].notna()
        for col in available_av_columns:
            mask = mask & ((self.data[col].notna()) & (self.data[col] != 0))
        
        clean_data = self.data[mask][['h_ip', 'param_combo', 'fp', 'cde', 'cdi'] + available_av_columns].copy()
        
        if clean_data.empty:
            print("No valid avalanche data found (all values are NaN or zero)!")
            return
        
        # Separate baseline and modified parameter data
        baseline_mask = (clean_data['fp'] == 0) & (clean_data['cde'] == 0) & (clean_data['cdi'] == 0)
        baseline_data = clean_data[baseline_mask]
        modified_data = clean_data[~baseline_mask]
        
        # Apply h_ip range filter for fitting if specified
        if fit_h_ip_range is not None:
            fit_baseline_data = baseline_data[
                (baseline_data['h_ip'] >= fit_h_ip_range[0]) & 
                (baseline_data['h_ip'] <= fit_h_ip_range[1])
            ]
            print(f"Fitting baseline data only for h_ip in range [{fit_h_ip_range[0]}, {fit_h_ip_range[1]}]")
            print(f"Using {len(fit_baseline_data)} out of {len(baseline_data)} baseline points for fitting")
        else:
            fit_baseline_data = baseline_data
            print(f"Using all {len(baseline_data)} baseline points for fitting")
        
        print(f"Total: {len(baseline_data)} baseline and {len(modified_data)} modified parameter data points")
        
        # Create color map for different parameter combinations (matching h_ip plots exactly)
        # Get ALL unique combinations from the entire dataset
        all_unique_combos = clean_data['param_combo'].unique()
        colors = {}
        markers = {}
        marker_list = ['s', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8']
        
        # Helper function to check if a combination is baseline
        def is_baseline(combo_str):
            parts = combo_str.split('_')
            fp_val = float(parts[0].split('=')[1])
            cde_val = float(parts[1].split('=')[1])
            cdi_val = float(parts[2].split('=')[1])
            return fp_val == 0.0 and cde_val == 0.0 and cdi_val == 0.0
        
        baseline_combo = None
        # Sort and assign colors/markers to ALL combinations
        for i, combo in enumerate(sorted(all_unique_combos)):
            if is_baseline(combo):
                colors[combo] = 'black'
                markers[combo] = 'o'
                baseline_combo = combo
            else:
                # Use Set1 colormap for more distinct colors (matching h_ip plots)
                colors[combo] = plt.cm.Set1(i % 9)  # Set1 has 9 distinct colors
                markers[combo] = marker_list[i % len(marker_list)]
        
        with PdfPages(save_path) as pdf:
            # Create individual plots for each avalanche metric
            n_available = len(available_av_columns)
            fig, axes = plt.subplots(1, n_available, figsize=(7*n_available, 7))
            if n_available == 1:
                axes = [axes]  # Make it a list for consistency
            fig.suptitle('Avalanche Scaling Analysis vs h_ip', fontsize=16, y=0.98)
            
            # Define metrics and their properties - only for available columns
            all_metrics_info = {
                'AV_Alpha': ('AVsize Exponent (α)', 'black'),
                'AV_Beta': ('AVduration Exponent (β)', 'black'),
                'AV_Scaling_Diff': ('Scaling Difference |σ - σ_fit|', 'black')
            }
            
            metrics_info = []
            for i, col in enumerate(available_av_columns):
                title, _ = all_metrics_info[col]
                metrics_info.append((col, title, axes[i]))
            
            # Store fit results
            fit_results = {}
            
            for metric, title, ax in metrics_info:
                # Plot baseline data and fit line
                if len(fit_baseline_data) > 0:
                    x_fit_data = np.log10(fit_baseline_data['h_ip'].values)
                    y_fit_data = fit_baseline_data[metric].values
                    
                    # Perform linear regression on filtered baseline data
                    if len(x_fit_data) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit_data, y_fit_data)
                        
                        # Store fit results
                        fit_results[metric] = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'std_err': std_err,
                            'n_points': len(x_fit_data),
                            'h_ip_range': fit_h_ip_range if fit_h_ip_range else (fit_baseline_data['h_ip'].min(), fit_baseline_data['h_ip'].max())
                        }
                        
                        # Create fit line that extends beyond all data points
                        all_x = np.log10(clean_data['h_ip'].values)
                        x_range = all_x.max() - all_x.min()
                        x_extension = x_range * 0.2  # Extend by 20% on each side
                        x_line = np.linspace(all_x.min() - x_extension, all_x.max() + x_extension, 100)
                        y_line = slope * x_line + intercept
                        
                        # Plot baseline fit line
                        ax.plot(10**x_line, y_line, color='black', linewidth=2.5, linestyle='-', 
                            alpha=0.8, zorder=3,
                            label=f'Baseline fit: y = {slope:.3f}·log(h_ip) + {intercept:.3f}')
                        
                        # Add shaded region showing the fitting range
                        if fit_h_ip_range is not None:
                            ax.axvspan(fit_h_ip_range[0], fit_h_ip_range[1], alpha=0.1, color='green', 
                                    label='Fitting range', zorder=1)
                        
                        # Add fit statistics to plot
                        if fit_h_ip_range is not None:
                            range_str = f'h_ip ∈ [{fit_h_ip_range[0]:.3f}, {fit_h_ip_range[1]:.3f}]'
                        else:
                            range_str = 'All baseline points'
                        
                        textstr = f'Baseline fit ({range_str}):\nR² = {r_value**2:.3f}\np = {p_value:.3e}\nn = {len(x_fit_data)} points'
                        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    # Plot ALL baseline data points with h_ip labels
                    if fit_h_ip_range is not None and len(baseline_data) > len(fit_baseline_data):
                        # Plot points outside fitting range in gray
                        out_of_range = baseline_data[
                            (baseline_data['h_ip'] < fit_h_ip_range[0]) | 
                            (baseline_data['h_ip'] > fit_h_ip_range[1])
                        ]
                        if len(out_of_range) > 0:
                            ax.scatter(out_of_range['h_ip'], out_of_range[metric], 
                                    color='gray', alpha=0.5, s=100, edgecolor='white', 
                                    linewidth=2, zorder=5, label='Baseline (excluded from fit)')
                            # Add h_ip labels
                            for _, row in out_of_range.iterrows():
                                ax.annotate(f'{row["h_ip"]:.3f}', 
                                           (row['h_ip'], row[metric]),
                                           xytext=(2, 2), textcoords='offset points',
                                           fontsize=6, alpha=0.7, color='gray')
                        
                        # Plot points within fitting range in black
                        ax.scatter(fit_baseline_data['h_ip'], fit_baseline_data[metric], 
                                color='black', alpha=0.8, s=120, edgecolor='white', 
                                linewidth=2, zorder=6, label='Baseline (fitted)')
                        # Add h_ip labels
                        for _, row in fit_baseline_data.iterrows():
                            ax.annotate(f'{row["h_ip"]:.3f}', 
                                       (row['h_ip'], row[metric]),
                                       xytext=(2, 2), textcoords='offset points',
                                       fontsize=6, alpha=0.9, color='black')
                    else:
                        # Plot all baseline points
                        ax.scatter(baseline_data['h_ip'], baseline_data[metric], 
                                color='black', alpha=0.8, s=120, edgecolor='white', 
                                linewidth=2, zorder=6, label='Baseline (fp=cde=cdi=0)')
                        # Add h_ip labels
                        for _, row in baseline_data.iterrows():
                            ax.annotate(f'{row["h_ip"]:.3f}', 
                                       (row['h_ip'], row[metric]),
                                       xytext=(2, 2), textcoords='offset points',
                                       fontsize=6, alpha=0.9, color='black')
                
                # Plot modified parameter data points with consistent style
                if len(modified_data) > 0:
                    # Sort unique combos for consistent ordering
                    unique_combos = sorted(modified_data['param_combo'].unique())
                    
                    for combo in unique_combos:
                        combo_data = modified_data[modified_data['param_combo'] == combo]
                        if len(combo_data) > 0:
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
                            label = ', '.join(label_parts)
                            
                            # Plot data points with consistent color and marker
                            ax.scatter(combo_data['h_ip'], combo_data[metric],
                                    color=colors[combo], marker=markers[combo], alpha=0.7, s=100, 
                                    edgecolor='black', linewidth=1, zorder=5, label=label)
                            
                            # Add h_ip labels with matching color
                            for _, row in combo_data.iterrows():
                                ax.annotate(f'{row["h_ip"]:.3f}', 
                                           (row['h_ip'], row[metric]),
                                           xytext=(2, 2), textcoords='offset points',
                                           fontsize=6, alpha=0.8, color=colors[combo])
                
                # Formatting
                ax.set_xlabel('h_ip', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(title, fontsize=14)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                
                # Create custom legend with reasonable number of entries
                handles, labels = ax.get_legend_handles_labels()
                # Put baseline and fitting-related entries first
                if len(handles) > 0:
                    baseline_indices = [i for i, l in enumerate(labels) if 'Baseline' in l or 'Fitting range' in l]
                    other_indices = [i for i in range(len(labels)) if i not in baseline_indices]
                    
                    handles = [handles[i] for i in baseline_indices] + [handles[i] for i in other_indices]
                    labels = [labels[i] for i in baseline_indices] + [labels[i] for i in other_indices]
                    
                    # Limit legend entries if too many
                    if len(handles) > 10:
                        ax.legend(handles[:10], labels[:10], loc='best', fontsize=8)
                    else:
                        ax.legend(loc='best', fontsize=8)
                
                # Set x-axis to show all h_ip values
                all_h_ips = sorted(clean_data['h_ip'].unique())
                ax.set_xticks(all_h_ips)
                ax.set_xticklabels([f'{x:.4g}' for x in all_h_ips])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # NEW PLOT: AVsize vs AVduration with consistent styling
            if 'AV_Alpha' in available_av_columns and 'AV_Beta' in available_av_columns:
                fig, ax = plt.subplots(figsize=(8, 8))
                fig.suptitle('AVsize vs AVduration Exponents', fontsize=16)
                
                # Plot baseline data and fit
                if len(fit_baseline_data) > 0:
                    x_fit = fit_baseline_data['AV_Alpha'].values
                    y_fit = fit_baseline_data['AV_Beta'].values
                    
                    # Perform linear regression on alpha vs beta
                    if len(x_fit) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                        
                        # Create fit line
                        x_range_alpha = clean_data['AV_Alpha'].max() - clean_data['AV_Alpha'].min()
                        x_extension = x_range_alpha * 0.2
                        x_line = np.linspace(clean_data['AV_Alpha'].min() - x_extension, 
                                            clean_data['AV_Alpha'].max() + x_extension, 100)
                        y_line = slope * x_line + intercept
                        
                        # Plot fit line
                        ax.plot(x_line, y_line, color='black', linewidth=2.5, linestyle='-', 
                            alpha=0.8, zorder=3,
                            label=f'Baseline fit: β = {slope:.3f}·α + {intercept:.3f}')
                        
                        # Add fit statistics
                        if fit_h_ip_range is not None:
                            range_str = f'h_ip ∈ [{fit_h_ip_range[0]:.3f}, {fit_h_ip_range[1]:.3f}]'
                        else:
                            range_str = 'All baseline points'
                        
                        textstr = f'Baseline fit ({range_str}):\nR² = {r_value**2:.3f}\np = {p_value:.3e}\nn = {len(x_fit)} points'
                        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    # Plot baseline points with h_ip labels
                    if fit_h_ip_range is not None and len(baseline_data) > len(fit_baseline_data):
                        # Plot points outside fitting range in gray
                        out_of_range = baseline_data[
                            (baseline_data['h_ip'] < fit_h_ip_range[0]) | 
                            (baseline_data['h_ip'] > fit_h_ip_range[1])
                        ]
                        if len(out_of_range) > 0:
                            ax.scatter(out_of_range['AV_Alpha'], out_of_range['AV_Beta'], 
                                    color='gray', alpha=0.5, s=120, edgecolor='white', 
                                    linewidth=2, zorder=5, label='Baseline (excluded from fit)')
                            # Add h_ip labels
                            for _, row in out_of_range.iterrows():
                                ax.annotate(f'{row["h_ip"]:.3f}', 
                                           (row['AV_Alpha'], row['AV_Beta']),
                                           xytext=(2, 2), textcoords='offset points',
                                           fontsize=6, alpha=0.7, color='gray')
                        
                        # Plot points within fitting range in black
                        ax.scatter(fit_baseline_data['AV_Alpha'], fit_baseline_data['AV_Beta'], 
                                color='black', alpha=0.8, s=140, edgecolor='white', 
                                linewidth=2, zorder=6, label='Baseline (fitted)')
                        # Add h_ip labels
                        for _, row in fit_baseline_data.iterrows():
                            ax.annotate(f'{row["h_ip"]:.3f}', 
                                       (row['AV_Alpha'], row['AV_Beta']),
                                       xytext=(2, 2), textcoords='offset points',
                                       fontsize=6, alpha=0.9, color='black')
                    else:
                        # Plot all baseline points
                        ax.scatter(baseline_data['AV_Alpha'], baseline_data['AV_Beta'], 
                                color='black', alpha=0.8, s=140, edgecolor='white', 
                                linewidth=2, zorder=6, label='Baseline')
                        # Add h_ip labels
                        for _, row in baseline_data.iterrows():
                            ax.annotate(f'{row["h_ip"]:.3f}', 
                                       (row['AV_Alpha'], row['AV_Beta']),
                                       xytext=(2, 2), textcoords='offset points',
                                       fontsize=6, alpha=0.9, color='black')
                
                # Plot modified parameter data with consistent styling
                if len(modified_data) > 0:
                    for combo in sorted(modified_data['param_combo'].unique()):
                        combo_data = modified_data[modified_data['param_combo'] == combo]
                        if len(combo_data) > 0:
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
                            label = ', '.join(label_parts)
                            
                            ax.scatter(combo_data['AV_Alpha'], combo_data['AV_Beta'],
                                    color=colors[combo], marker=markers[combo], alpha=0.7, s=100, 
                                    edgecolor='black', linewidth=1, zorder=5, label=label)
                            
                            # Add h_ip labels with matching color
                            for _, row in combo_data.iterrows():
                                ax.annotate(f'{row["h_ip"]:.3f}', 
                                           (row['AV_Alpha'], row['AV_Beta']),
                                           xytext=(2, 2), textcoords='offset points',
                                           fontsize=6, alpha=0.8, color=colors[combo])
                
                # Add theoretical predictions if relevant
                ax.axhline(y=2, color='red', linestyle=':', alpha=0.5, label='β = 2 (mean-field)')
                ax.axvline(x=1.5, color='red', linestyle=':', alpha=0.5, label='α = 1.5 (mean-field)')
                
                # Formatting
                ax.set_xlabel('AVsize Exponent (α)', fontsize=12)
                ax.set_ylabel('AVduration Exponent (β)', fontsize=12)
                ax.set_title('Relationship between AVsize and AVduration Exponents', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                
                # Set equal aspect ratio to better see the relationship
                ax.set_aspect('equal', adjustable='box')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
        # Save fit results to text file (only for baseline)
        fit_results_path = self.output_dir / 'avalanche_scaling_fit_results.txt'
        with open(fit_results_path, 'w', encoding='utf-8') as f:
            f.write("AVALANCHE SCALING FIT RESULTS (BASELINE ONLY)\n")
            f.write("=" * 60 + "\n\n")
            
            if fit_h_ip_range is not None:
                f.write(f"Fitting range: h_ip ∈ [{fit_h_ip_range[0]}, {fit_h_ip_range[1]}]\n")
            else:
                f.write("Fitting range: All baseline data points\n")
            f.write("\nLinear fits for baseline parameters (fp=cde=cdi=0):\n")
            f.write("metric = slope * log10(h_ip) + intercept\n\n")
            
            if fit_results:
                for metric in available_av_columns:
                    if metric in fit_results:
                        title = all_metrics_info[metric][0]
                        f.write(f"{title}:\n")
                        f.write(f"  Slope:     {fit_results[metric]['slope']:.6f} ± {fit_results[metric]['std_err']:.6f}\n")
                        f.write(f"  Intercept: {fit_results[metric]['intercept']:.6f}\n")
                        f.write(f"  R²:        {fit_results[metric]['r_squared']:.6f}\n")
                        f.write(f"  p-value:   {fit_results[metric]['p_value']:.6e}\n")
                        f.write(f"  N points:  {fit_results[metric]['n_points']}\n")
                        f.write(f"  h_ip range: [{fit_results[metric]['h_ip_range'][0]:.4f}, {fit_results[metric]['h_ip_range'][1]:.4f}]\n\n")
                
                # Add alpha vs beta relationship if computed
                if 'AV_Alpha' in available_av_columns and 'AV_Beta' in available_av_columns and len(fit_baseline_data) >= 2:
                    f.write("\nAVsize vs AVduration Relationship:\n")
                    f.write("-" * 40 + "\n")
                    f.write("β = slope * α + intercept\n")
                    # The slope and other stats would need to be stored from the alpha-beta fit above
                    # For now, we'll note that this relationship exists in the plot
                    f.write("See plot for fitted relationship between α and β\n\n")
                
                # Add interpretation
                f.write("INTERPRETATION (BASELINE TREND):\n")
                f.write("-" * 40 + "\n")
                
                # Check if alpha increases with h_ip
                if 'AV_Alpha' in fit_results:
                    alpha_slope = fit_results['AV_Alpha']['slope']
                    if alpha_slope > 0:
                        f.write("- AVsize exponent (α) increases with h_ip\n")
                        f.write("  → Avalanche size distributions become less heavy-tailed at higher h_ip\n")
                    else:
                        f.write("- AVsize exponent (α) decreases with h_ip\n")
                        f.write("  → Avalanche size distributions become more heavy-tailed at higher h_ip\n")
                
                # Check if beta increases with h_ip
                if 'AV_Beta' in fit_results:
                    beta_slope = fit_results['AV_Beta']['slope']
                    if beta_slope > 0:
                        f.write("- AVduration exponent (β) increases with h_ip\n")
                        f.write("  → Avalanche duration distributions become less heavy-tailed at higher h_ip\n")
                    else:
                        f.write("- AVduration exponent (β) decreases with h_ip\n")
                        f.write("  → Avalanche duration distributions become more heavy-tailed at higher h_ip\n")
                
                # Check scaling difference trend
                if 'AV_Scaling_Diff' in fit_results:
                    diff_slope = fit_results['AV_Scaling_Diff']['slope']
                    if abs(diff_slope) < 0.1:
                        f.write("- Scaling difference remains relatively constant with h_ip\n")
                        f.write("  → Scaling relation consistency across h_ip values\n")
                    elif diff_slope > 0:
                        f.write("- Scaling difference increases with h_ip\n")
                        f.write("  → Deviation from theoretical scaling increases at higher h_ip\n")
                    else:
                        f.write("- Scaling difference decreases with h_ip\n")
                        f.write("  → Better agreement with theoretical scaling at higher h_ip\n")
            else:
                f.write("No baseline data available for fitting.\n")
        
        print(f"Avalanche scaling plots saved to {save_path}")
        print(f"Fit results saved to {fit_results_path}")
            
        
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
        FIXED_WIDTH_PERCENT = 0.05
        
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
                colors[combo] = plt.cm.Set1(i % 40)
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
                        # Plot the line and markers
                        ax.plot(baseline_data['h_ip'], baseline_data[metric],
                                'ko-', markersize=3, linewidth=0.8,
                                alpha=0.9, zorder=10, markeredgewidth=0.5, markeredgecolor='white')
                        
                        # Add error rectangles if std data exists
                        if hasattr(self, 'std_columns') and metric in self.std_columns:
                            yerr_col = self.std_columns[metric]
                            if yerr_col in baseline_data.columns and baseline_data[yerr_col].notna().any():
                                # Create semi-transparent rectangles for error regions
                                x_values = baseline_data['h_ip'].values
                                y_values = baseline_data[metric].values
                                y_errors = baseline_data[yerr_col].values
                                
                                # For log scale x-axis, we need to calculate rectangle widths carefully
                                # Fixed width rectangles
                                for i in range(len(x_values)):
                                    if not np.isnan(y_errors[i]):
                                        x_center = x_values[i]
                                        # For log scale, we calculate left and right based on multiplicative factors
                                        x_left = x_center * (1 - FIXED_WIDTH_PERCENT)
                                        x_right = x_center * (1 + FIXED_WIDTH_PERCENT)
                                        
                                        # Draw rectangle
                                        rect = plt.Rectangle((x_left, y_values[i] - y_errors[i]),
                                                        x_right - x_left,
                                                        2 * y_errors[i],
                                                        facecolor='black', alpha=0.1,
                                                        edgecolor='none', zorder=5)
                                        ax.add_patch(rect)

                # Plot all other combinations with enhanced styling
                for combo in sorted(unique_combos):
                    if baseline_combo and combo == baseline_combo:
                        continue
                        
                    combo_data = self.data[self.data['param_combo'] == combo].sort_values('h_ip')
                    if not combo_data.empty:
                        # Plot the line and markers
                        ax.plot(combo_data['h_ip'], combo_data[metric],
                                marker=markers[combo], linestyle='--',
                                color=colors[combo],
                                markersize=3, linewidth=0.6, alpha=0.85,
                                markeredgewidth=0.4, markeredgecolor='white')
                        
                        # Add error rectangles if std data exists
                        if hasattr(self, 'std_columns') and metric in self.std_columns:
                            yerr_col = self.std_columns[metric]
                            if yerr_col in combo_data.columns and combo_data[yerr_col].notna().any():
                                # Create semi-transparent rectangles for error regions
                                x_values = combo_data['h_ip'].values
                                y_values = combo_data[metric].values
                                y_errors = combo_data[yerr_col].values
                                
                                # For log scale x-axis, we need to calculate rectangle widths carefully
                                for i in range(len(x_values)):
                                    if not np.isnan(y_errors[i]):
                                        x_center = x_values[i]
                                        # For log scale, we calculate left and right based on multiplicative factors
                                        x_left = x_center * (1 - FIXED_WIDTH_PERCENT)
                                        x_right = x_center * (1 + FIXED_WIDTH_PERCENT)
                                        
                                        # Draw rectangle with combo color
                                        rect = plt.Rectangle((x_left, y_values[i] - y_errors[i]),
                                                        x_right - x_left,
                                                        2 * y_errors[i],
                                                        facecolor=colors[combo], alpha=0.1,
                                                        edgecolor='none', zorder=4)
                                        ax.add_patch(rect)
                
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
        
        # Continue with Page 2 code...
            
            
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
            
            # Create visualizations
            self.create_color_coded_hip_plots()
            # Create avalanche scaling plots
            self.create_avalanche_scaling_plots()
            # Create a summary statistics file
            self.create_summary_statistics()
            
            print("\nAnalysis complete!")
            print("=" * 60)
            print("\nGenerated files:")
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
    base_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\finegrain"
    
    # Create analyzer instance
    analyzer = NeuralNetworkParameterAnalyzer(base_path)
    
    # Run the complete analysis
    analyzer.create_interactive_dashboard()
    
    print("\nTo use this tool with different data:")
    print("1. Update the 'base_path' variable to point to your test_single directory")
    print("2. Run the script to generate all visualizations")
    print("3. Check the generated PDF files for comprehensive analysis")