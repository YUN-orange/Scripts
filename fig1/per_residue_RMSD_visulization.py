#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HRSV Epitope Conservation Analysis Script
================================================================================

Description:
This script analyzes conservation patterns of antigenic epitopes in HRSV F protein
based on RMSD values from structural alignments. It generates visualizations 
showing residue-level conservation with antigenic site annotations.

Functionality:
1. Loads residue RMSD contribution data
2. Calculates conservation statistics (mean, standard deviation, CV)
3. Identifies known antigenic epitope regions
4. Generates bubble plot visualization with epitope annotations
5. Provides epitope-specific conservation analysis

Input Requirements:
1. CSV file with residue RMSD contributions (from first analysis script)
   Required columns: query, target, residue_number, rmsd_contribution

Output:
1. conservation_analysis_with_epitopes.png: Visualization plot
2. epitope_conservation_analysis.csv: Statistical analysis of epitopes

Usage:
1. Install dependencies: pip install pandas numpy matplotlib
2. Configure paths below (INPUT_FILE, OUTPUT_DIR, EXPECTED_MAX_RESIDUE)
3. Run: python this_script.py

Dependencies:
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0

Author: [Your Name/Affiliation]
Contact: [Your Email]
Date: [Date]
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION - USER MUST MODIFY THESE SETTINGS
# ============================================================================

# Path to input CSV file with RMSD contributions
INPUT_FILE = '/path/to/your/rmsd_contributions_merged.csv'

# Output directory for results
OUTPUT_DIR = '/path/to/output/directory/'

# Maximum expected residue number for filtering
EXPECTED_MAX_RESIDUE = 1461

# Known antigenic epitope regions for HRSV F protein
EPITOPE_REGIONS = {
    'Site Ø': [(1012, 1023), (1149, 1162)],
    'Site V': [(1004, 1011), (1115, 1128), (1141, 1146)],
    'Site I': [(330, 340)],
    'Site II': [(207, 227)],
    'Site IV': [(827, 843)],
    'Site III': [(473, 482), (711, 719), (749, 754)]
}

# Color scheme for epitope visualization
EPITOPE_COLORS = {
    'Site Ø': 'lightcoral',
    'Site V': 'lightblue',
    'Site I': 'lightgreen',
    'Site II': 'gold',
    'Site IV': 'plum',
    'Site III': 'orange'
}

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def load_and_filter_data():
    """
    Load RMSD contribution data and filter by residue number
    
    Returns:
    --------
    pd.DataFrame
        Filtered data with residue numbers within expected range
    """
    print("Loading and filtering data...")
    
    # Read data
    df = pd.read_csv(INPUT_FILE)
    print(f"Original data rows: {len(df)}")
    
    # Filter residues beyond expected range
    filtered_df = df[df['residue_number'] <= EXPECTED_MAX_RESIDUE]
    print(f"After filtering: {len(filtered_df)} rows")
    
    # Calculate filtering statistics
    total_filtered = len(df) - len(filtered_df)
    print(f"Total filtered out: {total_filtered} rows ({total_filtered/len(df):.2%})")
    print(f"Final residue range: {filtered_df['residue_number'].min()} - "
          f"{filtered_df['residue_number'].max()}")
    
    return filtered_df


def add_epitope_annotations(ax, y_min, y_max):
    """
    Add epitope region annotations to plot as background rectangles
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add annotations to
    y_min : float
        Minimum y-value for rectangle height
    y_max : float
        Maximum y-value for rectangle height
    """
    for epitope_name, regions in EPITOPE_REGIONS.items():
        color = EPITOPE_COLORS[epitope_name]
        
        for region in regions:
            if len(region) == 2:  # Continuous region
                start, end = region
                width = end - start + 1
                rect = Rectangle((start - 0.5, y_min), width, y_max - y_min,
                                alpha=0.2, color=color)
                ax.add_patch(rect)


def analyze_epitope_conservation(conservation_data):
    """
    Calculate conservation statistics for each epitope region
    
    Parameters:
    -----------
    conservation_data : pd.DataFrame
        DataFrame with conservation statistics per residue
    
    Returns:
    --------
    dict
        Dictionary with epitope conservation statistics
    """
    epitope_analysis = {}
    
    for epitope_name, regions in EPITOPE_REGIONS.items():
        epitope_residues = []
        
        # Collect all residues in this epitope
        for region in regions:
            if len(region) == 2:  # Continuous region
                start, end = region
                epitope_residues.extend(range(start, end + 1))
            else:  # Single residue
                epitope_residues.append(region[0])
        
        # Extract conservation data for epitope residues
        epitope_data = conservation_data[conservation_data.index.isin(epitope_residues)]
        
        if len(epitope_data) > 0:
            epitope_analysis[epitope_name] = {
                'residue_count': len(epitope_data),
                'mean_rmsd': epitope_data['mean'].mean(),
                'std_rmsd': epitope_data['std'].mean(),
                'cv': epitope_data['cv'].mean(),
                'conservation_rank': epitope_data['mean'].mean()  # Lower = more conserved
            }
    
    # Sort by conservation (lowest RMSD mean first)
    epitope_analysis = dict(sorted(epitope_analysis.items(),
                                 key=lambda x: x[1]['mean_rmsd']))
    
    return epitope_analysis


def add_bubble_size_legend(ax, bubble_sizes, conservation_std, min_size, max_size):
    """
    Add legend explaining bubble size mapping to standard deviation
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add legend to
    bubble_sizes : array-like
        Actual bubble sizes used in plot
    conservation_std : pd.Series
        Standard deviation values
    min_size : int
        Minimum bubble size
    max_size : int
        Maximum bubble size
    
    Returns:
    --------
    matplotlib.legend.Legend or None
        Bubble size legend if created, None otherwise
    """
    # Select representative bubble sizes for legend
    size_steps = [min_size, (min_size + max_size) // 2, max_size]
    
    # Calculate standard deviation range
    size_range = max_size - min_size
    std_range = conservation_std.max() - conservation_std.min()
    
    if size_range > 0 and std_range > 0:
        # Calculate std values corresponding to each bubble size
        std_values = []
        for size in size_steps:
            std_value = conservation_std.min() + (size - min_size) / size_range * std_range
            std_values.append(std_value)
        
        # Create legend elements
        legend_elements = []
        for size, std_val in zip(size_steps, std_values):
            scatter_handle = plt.scatter([], [], s=size, color='gray', alpha=0.7,
                                       edgecolors='black', linewidth=0.5)
            legend_elements.append((scatter_handle, f'Std={std_val:.2f}Å'))
        
        # Extract handles and labels
        legend_handles = [elem[0] for elem in legend_elements]
        legend_labels = [elem[1] for elem in legend_elements]
        
        # Create legend
        bubble_legend = ax.legend(legend_handles, legend_labels,
                                loc='upper left', fontsize=21,
                                title='Bubble Size (Std Dev)', title_fontsize=24)
        return bubble_legend
    
    return None


def calculate_conservation_statistics(filtered_data):
    """
    Calculate conservation statistics from RMSD contribution data
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Filtered RMSD contribution data
    
    Returns:
    --------
    tuple
        (conservation DataFrame, global mean, epitope analysis dictionary)
    """
    # Initialize statistics dictionaries
    sums = {}
    squares = {}
    counts = {}
    
    # Process data in chunks for memory efficiency
    chunk_size = 100000
    total_chunks = (len(filtered_data) - 1) // chunk_size + 1
    
    for i in range(0, len(filtered_data), chunk_size):
        chunk = filtered_data.iloc[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{total_chunks}")
        
        # Remove missing values
        chunk = chunk.dropna(subset=['rmsd_contribution'])
        
        if len(chunk) == 0:
            continue
        
        # Calculate statistics for this chunk
        chunk_grouped = chunk.groupby('residue_number')['rmsd_contribution'].agg(['sum', 'count'])
        chunk_grouped['sum_of_squares'] = chunk.groupby('residue_number')['rmsd_contribution'].apply(
            lambda x: (x ** 2).sum())
        
        # Accumulate statistics
        for residue, row in chunk_grouped.iterrows():
            if residue not in sums:
                sums[residue] = 0
                squares[residue] = 0
                counts[residue] = 0
            
            sums[residue] += row['sum']
            squares[residue] += row['sum_of_squares']
            counts[residue] += row['count']
    
    # Calculate final statistics
    conservation_data = []
    for residue in sorted(sums.keys()):
        mean = sums[residue] / counts[residue]
        variance = (squares[residue] / counts[residue]) - (mean ** 2)
        std = np.sqrt(variance) if variance > 0 else 0
        cv = std / mean if mean != 0 else 0
        
        conservation_data.append({
            'residue_number': residue,
            'mean': mean,
            'std': std,
            'cv': cv
        })
    
    # Create DataFrame
    conservation = pd.DataFrame(conservation_data).set_index('residue_number')
    
    # Calculate global mean
    global_mean = sum(sums.values()) / sum(counts.values())
    
    # Analyze epitope conservation
    epitope_analysis = analyze_epitope_conservation(conservation)
    
    return conservation, global_mean, epitope_analysis


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_conservation_plot(conservation, global_mean, epitope_analysis, output_dir):
    """
    Create bubble plot visualization of residue conservation
    
    Parameters:
    -----------
    conservation : pd.DataFrame
        Conservation statistics DataFrame
    global_mean : float
        Global mean RMSD value
    epitope_analysis : dict
        Epitope conservation analysis results
    output_dir : str
        Directory to save output files
    """
    # Set plotting parameters
    plt.rcParams.update({
        'font.size': 36,
        'legend.fontsize': 27,
        'axes.titlesize': 36,
        'axes.labelsize': 31.5,
        'xtick.labelsize': 27,
        'ytick.labelsize': 27,
    })
    
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(24, 12), dpi=300)
    
    # Calculate bubble sizes based on standard deviation
    min_size = 30
    max_size = 450
    size_range = conservation['std'].max() - conservation['std'].min()
    
    if size_range > 0:
        bubble_sizes = min_size + (conservation['std'] - conservation['std'].min()) / size_range * (max_size - min_size)
    else:
        bubble_sizes = np.full_like(conservation['std'], (min_size + max_size) / 2)
    
    # Create scatter plot
    scatter = ax1.scatter(
        x=conservation.index,
        y=conservation['mean'],
        c=conservation['cv'],
        s=bubble_sizes,
        cmap='coolwarm',
        vmin=0.4,  # Set color scale limits
        vmax=1.0,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add epitope annotations
    y_min, y_max = ax1.get_ylim()
    add_epitope_annotations(ax1, y_min, y_max)
    
    # Add global mean reference line
    global_mean_line = ax1.axhline(
        y=global_mean,
        color='red',
        linestyle='--',
        linewidth=3,
        alpha=0.7,
        label=f'Global Mean ({global_mean:.2f}Å)'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.01)
    cbar.set_label('Coefficient of Variation (CV)', rotation=270, labelpad=30, fontsize=31.5)
    cbar.ax.tick_params(labelsize=27)
    
    # Add bubble size legend
    bubble_legend = add_bubble_size_legend(ax1, bubble_sizes, conservation['std'], min_size, max_size)
    
    # Add global mean legend
    global_legend = ax1.legend(handles=[global_mean_line], loc='upper right', fontsize=27)
    
    # Ensure both legends are displayed
    if bubble_legend is not None:
        ax1.add_artist(bubble_legend)
    
    # Set plot labels and title
    ax1.set_title(
        f'Residue Conservation Analysis with Antigenic Sites\n'
        'Color: Coefficient of Variation | Bubble Size: Standard Deviation',
        pad=20,
        fontsize=36,
        y=1.02
    )
    ax1.set_xlabel('Residue Number', fontsize=31.5)
    ax1.set_ylabel('Mean RMSD Contribution (Å)', fontsize=31.5)
    
    # Configure grid and limits
    ax1.grid(alpha=0.2, linestyle=':')
    ax1.set_xlim(0, EXPECTED_MAX_RESIDUE + 50)
    ax1.tick_params(axis='both', labelsize=27)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "conservation_analysis_with_epitopes.png")
    plt.savefig(output_path, bbox_inches='tight', transparent=False, dpi=300)
    plt.close()
    
    return output_path


def save_results(conservation, global_mean, epitope_analysis, output_dir):
    """
    Save analysis results to files
    
    Parameters:
    -----------
    conservation : pd.DataFrame
        Conservation statistics
    global_mean : float
        Global mean RMSD
    epitope_analysis : dict
        Epitope analysis results
    output_dir : str
        Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save epitope analysis to CSV
    epitope_df = pd.DataFrame.from_dict(epitope_analysis, orient='index')
    epitope_output_path = os.path.join(output_dir, "epitope_conservation_analysis.csv")
    epitope_df.to_csv(epitope_output_path)
    
    # Print bubble size statistics
    bubble_std_stats = conservation['std'].describe()
    
    print(f"\nGlobal Mean: {global_mean:.4f}Å")
    print(f"\nBubble Size (Standard Deviation) Statistics:")
    print(f"  Minimum: {bubble_std_stats['min']:.4f}Å")
    print(f"  25th percentile: {bubble_std_stats['25%']:.4f}Å")
    print(f"  Median: {bubble_std_stats['50%']:.4f}Å")
    print(f"  75th percentile: {bubble_std_stats['75%']:.4f}Å")
    print(f"  Maximum: {bubble_std_stats['max']:.4f}Å")
    print(f"  Mean: {bubble_std_stats['mean']:.4f}Å")
    print(f"  Std of std: {conservation['std'].std():.4f}Å")
    
    return epitope_output_path


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to execute epitope conservation analysis
    """
    print("=" * 80)
    print("HRSV Epitope Conservation Analysis")
    print("=" * 80)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Please update the INPUT_FILE path in the script configuration.")
        return
    
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Maximum residue number: {EXPECTED_MAX_RESIDUE}")
    print("-" * 80)
    
    try:
        # Step 1: Load and filter data
        filtered_data = load_and_filter_data()
        
        if len(filtered_data) == 0:
            print("ERROR: No valid data after filtering.")
            return
        
        # Step 2: Calculate conservation statistics
        print("\nCalculating conservation statistics...")
        conservation, global_mean, epitope_analysis = calculate_conservation_statistics(filtered_data)
        
        # Print epitope analysis results
        print("\nEpitope Conservation Analysis:")
        print("-" * 60)
        for epitope, data in epitope_analysis.items():
            print(f"{epitope}: Mean RMSD = {data['mean_rmsd']:.3f}Å, "
                  f"Residues = {data['residue_count']}")
        
        # Step 3: Create visualization
        print("\nCreating visualization...")
        plot_path = create_conservation_plot(conservation, global_mean, epitope_analysis, OUTPUT_DIR)
        
        # Step 4: Save results
        epitope_path = save_results(conservation, global_mean, epitope_analysis, OUTPUT_DIR)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Unique residues analyzed: {len(conservation)}")
        print(f"Visualization saved to: {plot_path}")
        print(f"Epitope analysis saved to: {epitope_path}")
        
        # Print epitope conservation ranking
        print("\nEpitope Conservation Ranking (most to least conserved):")
        print("-" * 60)
        for i, (epitope, data) in enumerate(epitope_analysis.items(), 1):
            print(f"{i}. {epitope}: {data['mean_rmsd']:.3f}Å")
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except pd.errors.EmptyDataError:
        print("ERROR: Input file is empty or incorrectly formatted.")
    except Exception as e:
        print(f"Unexpected error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()