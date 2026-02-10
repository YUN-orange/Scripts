#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRSV Antigenic Epitope RMSD Analysis Script

Description:
This script analyzes RMSD differences in antigenic epitopes between HRSV subtypes.
It takes FoldSeek alignment results and PDB structures as input, calculates per-residue
RMSD contributions, and performs statistical comparisons of epitope conservation
between HRSV_A and HRSV_B subtypes.

Author: Jie-mei Yu
Date: 2026/2/10

Requirements:
- Python 3.7+
- Biopython, pandas, numpy, matplotlib

Input Files:
1. FoldSeck alignment results in TSV format
2. PDB structure files

Output:
1. Per-residue RMSD contributions
2. Epitope-specific RMSD statistics
3. Comparative visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - PLEASE MODIFY THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Path to directory containing PDB files
PDB_DIR = "/path/to/your/pdb/files/"

# Path to FoldSeek alignment results (TSV format)
ALN_RESULTS = "/path/to/your/foldseek/results.tsv"

# Output directory for results
OUTPUT_DIR = "/path/to/output/directory/"

# Maximum expected residue number (used for filtering)
EXPECTED_MAX_RESIDUE = 1461

# Known antigenic epitope regions for HRSV F protein
EPITOPE_REGIONS = {
    'Site Ø': [(1012, 1023), (1149, 1162)],  # Discontinuous regions
    'Site V': [(1004, 1011), (1115, 1128), (1141, 1146)],
    'Site I': [(355, 375)],
    'Site II': [(255, 275)],
    'Site IV': [(884, 900)],
    'Site III': [(507, 516), (766, 774), (806, 811)]
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

# Column names for FoldSeek TSV output
COLUMN_NAMES = ['query', 'target', 'qstart', 'qend', 'qaln', 'taln', 'evalue', 'rmsd']

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_virus_type(pdb_id):
    """
    Determine virus subtype from PDB ID
    
    Parameters:
    -----------
    pdb_id : str
        PDB identifier
    
    Returns:
    --------
    str : Virus subtype (HRSV_A, HRSV_B, HMPV, or UNKNOWN)
    """
    pdb_lower = str(pdb_id).lower()
    
    if any(pattern in pdb_lower for pattern in ['hrsv_a', 'rsv_a', '_a', 'a_']):
        return 'HRSV_A'
    elif any(pattern in pdb_lower for pattern in ['hrsv_b', 'rsv_b', '_b', 'b_']):
        return 'HRSV_B'
    elif any(pattern in pdb_lower for pattern in ['hmpv', 'metapneumovirus']):
        return 'HMPV'
    else:
        # Inference from naming patterns
        if pdb_lower.endswith('a') or 'a' in pdb_lower[-3:]:
            return 'HRSV_A'
        elif pdb_lower.endswith('b') or 'b' in pdb_lower[-3:]:
            return 'HRSV_B'
        else:
            return 'UNKNOWN'


def load_and_filter_data(input_file):
    """
    Load RMSD contribution data and filter by residue number
    
    Parameters:
    -----------
    input_file : str
        Path to RMSD contributions CSV file
    
    Returns:
    --------
    pd.DataFrame : Filtered data with virus type annotations
    """
    print("Loading data...")
    
    # Read data
    df = pd.read_csv(input_file)
    print(f"Original data rows: {len(df)}")
    
    # Filter residues beyond expected range
    filtered_df = df[df['residue_number'] <= EXPECTED_MAX_RESIDUE]
    print(f"After filtering: {len(filtered_df)} rows")
    
    # Add virus type annotations
    print("Adding virus type annotations...")
    
    # Collect all unique PDB IDs
    all_ids = set(filtered_df['query']).union(set(filtered_df['target']))
    
    # Create type mapping
    type_dict = {}
    for pdb_id in all_ids:
        type_dict[pdb_id] = get_virus_type(pdb_id)
    
    # Add type columns
    filtered_df['query_type'] = filtered_df['query'].map(type_dict)
    filtered_df['target_type'] = filtered_df['target'].map(type_dict)
    
    # Report type distribution
    print("\nVirus type distribution:")
    print("Query types:", filtered_df['query_type'].value_counts().to_dict())
    print("Target types:", filtered_df['target_type'].value_counts().to_dict())
    
    return filtered_df


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_epitope_statistics_by_comparison_type(filtered_data):
    """
    Calculate epitope RMSD statistics for different comparison types
    
    Comparison types:
    1. HRSV_A_internal: Within HRSV_A subtype comparisons
    2. HRSV_B_internal: Within HRSV_B subtype comparisons
    3. HRSV_A_vs_B: Between HRSV_A and HRSV_B subtypes
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Filtered RMSD contribution data
    
    Returns:
    --------
    dict : Dictionary containing statistics for each comparison type
    """
    print("\n" + "="*80)
    print("Calculating epitope RMSD statistics by comparison type")
    print("="*80)
    
    # Define comparison types
    comparison_types = {
        'HRSV_A_internal': ('HRSV_A', 'HRSV_A'),
        'HRSV_B_internal': ('HRSV_B', 'HRSV_B'),
        'HRSV_A_vs_B': ('HRSV_A', 'HRSV_B')
    }
    
    all_results = {}
    
    for comp_name, (query_type, target_type) in comparison_types.items():
        print(f"\nProcessing {comp_name}...")
        
        # Filter data for this comparison type
        if comp_name in ['HRSV_A_internal', 'HRSV_B_internal']:
            comp_data = filtered_data[
                (filtered_data['query_type'] == query_type) &
                (filtered_data['target_type'] == target_type)
            ]
        elif comp_name == 'HRSV_A_vs_B':
            # Include both A vs B and B vs A comparisons
            comp_data = filtered_data[
                ((filtered_data['query_type'] == 'HRSV_A') &
                 (filtered_data['target_type'] == 'HRSV_B')) |
                ((filtered_data['query_type'] == 'HRSV_B') &
                 (filtered_data['target_type'] == 'HRSV_A'))
            ]
        
        print(f"  Alignments found: {len(comp_data)}")
        
        if len(comp_data) == 0:
            print(f"  Warning: No data for {comp_name}, skipping...")
            continue
        
        # Group by alignment
        alignments = comp_data.groupby(['query', 'target'])
        
        # Calculate epitope RMSD for each alignment
        epitope_results = []
        
        for (query, target), group in alignments:
            # Create residue to RMSD mapping
            residue_rmsd = dict(zip(group['residue_number'], 
                                    group['rmsd_contribution']))
            
            # Calculate RMSD for each epitope
            for epitope_name, regions in EPITOPE_REGIONS.items():
                epitope_values = []
                
                for region in regions:
                    if len(region) == 2:
                        start, end = region
                        for res in range(start, end + 1):
                            if res in residue_rmsd:
                                epitope_values.append(residue_rmsd[res])
                
                if epitope_values:
                    epitope_rmsd = np.mean(epitope_values)
                    
                    epitope_results.append({
                        'comparison_type': comp_name,
                        'query': query,
                        'target': target,
                        'epitope': epitope_name,
                        'epitope_rmsd': epitope_rmsd,
                        'residue_count': len(epitope_values)
                    })
        
        # Convert to DataFrame
        epitope_by_alignment = pd.DataFrame(epitope_results)
        
        # Calculate summary statistics for each epitope
        epitope_stats = {}
        for epitope_name in EPITOPE_REGIONS.keys():
            epitope_data = epitope_by_alignment[
                epitope_by_alignment['epitope'] == epitope_name
            ]
            
            if len(epitope_data) > 0:
                mean_val = epitope_data['epitope_rmsd'].mean()
                std_val = epitope_data['epitope_rmsd'].std()
                sem_val = std_val / np.sqrt(len(epitope_data))
                
                epitope_stats[epitope_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'sem': sem_val,
                    'min': epitope_data['epitope_rmsd'].min(),
                    'max': epitope_data['epitope_rmsd'].max(),
                    'n_alignments': len(epitope_data),
                    'n_residues_avg': epitope_data['residue_count'].mean()
                }
            else:
                epitope_stats[epitope_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'sem': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'n_alignments': 0,
                    'n_residues_avg': 0
                }
        
        all_results[comp_name] = {
            'stats': epitope_stats,
            'details': epitope_by_alignment
        }
        
        # Print summary
        print(f"\n  {comp_name} epitope RMSD summary:")
        print("-" * 80)
        print(f"{'Epitope':<10} {'Mean (Å)':<10} {'SEM (Å)':<10} {'Alignments':<10}")
        print("-" * 80)
        for epitope, stats in epitope_stats.items():
            if not np.isnan(stats['mean']):
                print(f"{epitope:<10} {stats['mean']:<10.3f} "
                      f"{stats['sem']:<10.3f} {stats['n_alignments']:<10}")
    
    return all_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_plot(all_results, output_dir):
    """
    Create visualization comparing epitope RMSD across comparison types
    
    Parameters:
    -----------
    all_results : dict
        Results from calculate_epitope_statistics_by_comparison_type
    output_dir : str
        Directory to save output figures
    """
    print("\nCreating comparison visualization...")
    
    epitope_names = list(EPITOPE_REGIONS.keys())
    comparison_types = ['HRSV_A_internal', 'HRSV_B_internal', 'HRSV_A_vs_B']
    comparison_labels = ['Within HRSV-A', 'Within HRSV-B', 'HRSV-A vs HRSV-B']
    
    # Collect data for plotting
    mean_data = {}
    sem_data = {}
    
    for comp_name, comp_label in zip(comparison_types, comparison_labels):
        if comp_name in all_results:
            stats = all_results[comp_name]['stats']
            means = []
            sems = []
            
            for epitope in epitope_names:
                if epitope in stats and not np.isnan(stats[epitope]['mean']):
                    means.append(stats[epitope]['mean'])
                    sems.append(stats[epitope]['sem'])
                else:
                    means.append(0)
                    sems.append(0)
            
            mean_data[comp_label] = means
            sem_data[comp_label] = sems
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                            gridspec_kw={'height_ratios': [2, 1]})
    
    # Top panel: Bar plot of epitope RMSD
    ax1 = axes[0]
    x = np.arange(len(epitope_names))
    width = 0.25
    colors = ['steelblue', 'lightcoral', 'forestgreen']
    
    for i, (comp_label, color) in enumerate(zip(comparison_labels, colors)):
        if comp_label in mean_data:
            means = mean_data[comp_label]
            sems = sem_data[comp_label]
            
            bars = ax1.bar(x + i * width - width, means, width,
                          label=comp_label, color=color, alpha=0.8,
                          yerr=sems, capsize=5, error_kw={'elinewidth': 1.5})
            
            # Add value labels
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                            f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Antigenic Site', fontsize=12)
    ax1.set_ylabel('RMSD (Å)', fontsize=12)
    ax1.set_title('Epitope RMSD Comparison Across HRSV Subtypes', fontsize=14, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(epitope_names, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Bottom panel: Ratio plot (inter-subtype / intra-subtype)
    ax2 = axes[1]
    
    if 'Within HRSV-A' in mean_data and 'HRSV-A vs HRSV-B' in mean_data:
        ratios = []
        ratio_labels = []
        
        for i, epitope in enumerate(epitope_names):
            intra_mean = mean_data['Within HRSV-A'][i]
            inter_mean = mean_data['HRSV-A vs HRSV-B'][i]
            
            if intra_mean > 0 and inter_mean > 0:
                ratio = inter_mean / intra_mean
                ratios.append(ratio)
                ratio_labels.append(epitope)
        
        if ratios:
            x_ratio = np.arange(len(ratio_labels))
            bars = ax2.bar(x_ratio, ratios, color='gold', alpha=0.7)
            
            # Add value labels
            for bar, ratio_val in zip(bars, ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                        f'{ratio_val:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add reference line at ratio = 1
            ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            ax2.set_xlabel('Antigenic Site', fontsize=12)
            ax2.set_ylabel('Ratio (Inter/Intra)', fontsize=12)
            ax2.set_title('Inter-subtype to Intra-subtype RMSD Ratio', fontsize=12, pad=20)
            ax2.set_xticks(x_ratio)
            ax2.set_xticklabels(ratio_labels, rotation=45, ha='right', fontsize=11)
            ax2.grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "epitope_rmsd_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")
    return output_path


def create_epitope_summary_table(all_results, output_dir):
    """
    Create summary table of epitope RMSD statistics
    
    Parameters:
    -----------
    all_results : dict
        Analysis results
    output_dir : str
        Output directory
    
    Returns:
    --------
    str : Path to saved table
    """
    print("\nCreating summary table...")
    
    epitope_names = list(EPITOPE_REGIONS.keys())
    comparison_types = ['HRSV_A_internal', 'HRSV_B_internal', 'HRSV_A_vs_B']
    
    # Create summary DataFrame
    summary_data = []
    
    for epitope in epitope_names:
        row = {'Epitope': epitope}
        
        for comp_type in comparison_types:
            if comp_type in all_results:
                stats = all_results[comp_type]['stats'].get(epitope, {})
                
                if stats and not np.isnan(stats['mean']):
                    row[f'{comp_type}_mean'] = stats['mean']
                    row[f'{comp_type}_sem'] = stats['sem']
                    row[f'{comp_type}_n'] = stats['n_alignments']
                else:
                    row[f'{comp_type}_mean'] = 'N/A'
                    row[f'{comp_type}_sem'] = 'N/A'
                    row[f'{comp_type}_n'] = 0
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "epitope_rmsd_summary.csv")
    summary_df.to_csv(output_path, index=False)
    
    print(f"Summary table saved to: {output_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("EPITOPE RMSD SUMMARY TABLE (Mean ± SEM, Å)")
    print("="*80)
    print(f"{'Epitope':<10} {'Within HRSV-A':<20} {'Within HRSV-B':<20} "
          f"{'HRSV-A vs B':<20}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        a_str = (f"{row['HRSV_A_internal_mean']:.2f} ± "
                f"{row['HRSV_A_internal_sem']:.2f}" 
                if row['HRSV_A_internal_mean'] != 'N/A' else 'N/A')
        
        b_str = (f"{row['HRSV_B_internal_mean']:.2f} ± "
                f"{row['HRSV_B_internal_sem']:.2f}" 
                if row['HRSV_B_internal_mean'] != 'N/A' else 'N/A')
        
        ab_str = (f"{row['HRSV_A_vs_B_mean']:.2f} ± "
                 f"{row['HRSV_A_vs_B_sem']:.2f}" 
                 if row['HRSV_A_vs_B_mean'] != 'N/A' else 'N/A')
        
        print(f"{row['Epitope']:<10} {a_str:<20} {b_str:<20} {ab_str:<20}")
    
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("HRSV EPITOPE RMSD ANALYSIS")
    print("=" * 80)
    
    # Check if input files exist
    if not os.path.exists(ALN_RESULTS):
        print(f"ERROR: Alignment results file not found: {ALN_RESULTS}")
        print("Please update the ALN_RESULTS path in the script configuration.")
        return
    
    if not os.path.exists(PDB_DIR):
        print(f"WARNING: PDB directory not found: {PDB_DIR}")
        print("Please update the PDB_DIR path in the script configuration.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load and filter data
    filtered_data = load_and_filter_data(ALN_RESULTS)
    
    if len(filtered_data) == 0:
        print("ERROR: No valid data found after filtering.")
        return
    
    # Step 2: Calculate epitope statistics
    all_results = calculate_epitope_statistics_by_comparison_type(filtered_data)
    
    # Step 3: Create visualization
    plot_path = create_comparison_plot(all_results, OUTPUT_DIR)
    
    # Step 4: Create summary table
    table_path = create_epitope_summary_table(all_results, OUTPUT_DIR)
    
    # Step 5: Save detailed results
    print("\nSaving detailed results...")
    for comp_name, results in all_results.items():
        # Save statistics
        stats_df = pd.DataFrame.from_dict(results['stats'], orient='index')
        stats_path = os.path.join(OUTPUT_DIR, f"{comp_name}_statistics.csv")
        stats_df.to_csv(stats_path)
        print(f"  {comp_name} statistics saved to: {stats_path}")
        
        # Save alignment details
        details_path = os.path.join(OUTPUT_DIR, f"{comp_name}_details.csv")
        results['details'].to_csv(details_path, index=False)
        print(f"  {comp_name} details saved to: {details_path}")
    
    # Print completion message
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Main figure: {plot_path}")
    print(f"Summary table: {table_path}")
    print("\nFiles generated:")
    print("  • epitope_rmsd_comparison.png - Visualization figure")
    print("  • epitope_rmsd_summary.csv - Summary statistics")
    print("  • [comparison_type]_statistics.csv - Detailed statistics")
    print("  • [comparison_type]_details.csv - Alignment-level data")


if __name__ == "__main__":
    main()