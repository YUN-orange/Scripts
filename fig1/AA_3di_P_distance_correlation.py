#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Virus Sequence-Structure Distance Correlation Analysis Script
================================================================================

Description:
This script analyzes the correlation between amino acid sequence distance and
3di structural distance for multiple virus types. It generates scatter plots
comparing pairwise distances among six target virus types.

Functionality:
1. Loads aligned amino acid and 3di sequence files
2. Calculates pairwise distance matrices using p-distance
3. Identifies virus types from sequence names
4. Collects comparison data for all virus pairs
5. Generates scatter plot visualization with regression analysis

Input Requirements:
1. Aligned amino acid sequences in FASTA format
2. Aligned 3di structural sequences in FASTA format
3. Sequence names should contain virus type identifiers

Output:
1. all_virus_pairwise_comparisons_complete2.png: Scatter plot visualization
2. Console output with statistical analysis

Usage:
1. Install dependencies: pip install numpy matplotlib biopython scipy
2. Configure paths below (BASE_DIR, AA_FILE, DI_FILE)
3. Run: python this_script.py

Dependencies:
- numpy>=1.21.0
- matplotlib>=3.4.0
- biopython>=1.79
- scipy>=1.7.0

Author: [Your Name/Affiliation]
Contact: [Your Email]
Date: [Date]
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import os
from scipy import stats

# ============================================================================
# CONFIGURATION - USER MUST MODIFY THESE SETTINGS
# ============================================================================

# Base directory containing sequence files
BASE_DIR = "/path/to/your/sequence/directory"

# File names for aligned sequences
AA_FILE = "aa_aligned.fasta"  # Amino acid alignment
DI_FILE = "3di_aligned.fasta"  # 3di structural alignment

# Target virus types to analyze
TARGET_VIRUS_TYPES = ['HRSV_A', 'HRSV_B', 'BRSV', 'AMPV', 'HMPV', 'murinePV']

# Virus type identification patterns
VIRUS_PATTERNS = {
    'HRSV_A': ['HRSVA', 'RSVA', '_A.', '.A.'],
    'HRSV_B': ['HRSVB', 'RSVB', '_B.', '.B.'],
    'BRSV': ['BRSV'],
    'AMPV': ['AMPV'],
    'HMPV': ['HMPV'],
    'murinePV': ['micePV', 'MOUSE', 'MURINE']
}

# Visualization parameters
PLOT_FIGSIZE = (24, 14)
FONT_SIZES = {
    'label': 27,
    'title': 27,
    'tick': 24,
    'legend': 24
}

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def load_sequence_files(base_dir, aa_file, di_file):
    """
    Load amino acid and 3di alignment files
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing sequence files
    aa_file : str
        Amino acid alignment filename
    di_file : str
        3di structural alignment filename
    
    Returns:
    --------
    tuple
        (aa_alignment, di_alignment, success_flag)
    """
    # Construct full file paths
    aa_path = os.path.join(base_dir, aa_file)
    di_path = os.path.join(base_dir, di_file)
    
    # Check if files exist
    for f_path, f_name in [(aa_path, "Amino acid alignment"), 
                          (di_path, "3di structural alignment")]:
        if not os.path.exists(f_path):
            print(f"Error: {f_name} file not found - {f_path}")
            return None, None, False
    
    print("Reading alignment files...")
    
    try:
        # Read alignment files
        aa_alignment = AlignIO.read(aa_path, "fasta")
        di_alignment = AlignIO.read(di_path, "fasta")
        
        print(f"Reading complete:")
        print(f"  AA alignment: {len(aa_alignment)} sequences")
        print(f"  3di alignment: {len(di_alignment)} sequences")
        
        return aa_alignment, di_alignment, True
        
    except Exception as e:
        print(f"Error reading alignment files: {e}")
        return None, None, False


def identify_virus_types(sequence_names):
    """
    Identify virus types from sequence names
    
    Parameters:
    -----------
    sequence_names : list
        List of sequence identifiers
    
    Returns:
    --------
    tuple
        (virus_types dictionary, virus_counts dictionary)
    """
    print("\nIdentifying virus types from sequence names...")
    
    virus_types = {}
    virus_counts = {vtype: 0 for vtype in VIRUS_PATTERNS.keys()}
    virus_counts['Other'] = 0
    
    for name in sequence_names:
        name_upper = name.upper()  # Case-insensitive matching
        found = False
        
        # Check against known virus patterns
        for vtype, patterns in VIRUS_PATTERNS.items():
            for pattern in patterns:
                if pattern.upper() in name_upper:
                    virus_types[name] = vtype
                    virus_counts[vtype] += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            virus_types[name] = 'Other'
            virus_counts['Other'] += 1
    
    return virus_types, virus_counts


def calculate_distance_matrices(aa_alignment, di_alignment):
    """
    Calculate p-distance matrices for amino acid and 3di alignments
    
    Parameters:
    -----------
    aa_alignment : MultipleSeqAlignment
        Amino acid sequence alignment
    di_alignment : MultipleSeqAlignment
        3di structural alignment
    
    Returns:
    --------
    tuple
        (aa_distance_matrix, di_distance_matrix)
    """
    print("\nCalculating distance matrices...")
    calculator = DistanceCalculator('identity')  # p-distance
    
    # Calculate AA distance matrix
    print("  Calculating AA distance matrix...")
    aa_distance = calculator.get_distance(aa_alignment)
    aa_matrix = np.array([list(row) for row in aa_distance])
    
    # Calculate 3di distance matrix
    print("  Calculating 3di distance matrix...")
    di_distance = calculator.get_distance(di_alignment)
    di_matrix = np.array([list(row) for row in di_distance])
    
    return aa_matrix, di_matrix


def collect_comparison_data(taxa_names, virus_types, aa_matrix, di_matrix):
    """
    Collect comparison data for all virus pairs
    
    Parameters:
    -----------
    taxa_names : list
        List of sequence names
    virus_types : dict
        Virus type for each sequence
    aa_matrix : np.array
        Amino acid distance matrix
    di_matrix : np.array
        3di structural distance matrix
    
    Returns:
    --------
    dict
        Comparison data organized by virus pair type
    """
    print("\nCollecting comparison data...")
    
    # Initialize comparison data storage
    comparison_data = {}
    
    # Generate all possible virus pairs (including self-comparisons)
    all_possible_pairs = []
    for i, v1 in enumerate(TARGET_VIRUS_TYPES):
        for j in range(i, len(TARGET_VIRUS_TYPES)):
            v2 = TARGET_VIRUS_TYPES[j]
            
            if v1 == v2:
                key = f"{v1}_self"
            else:
                # Sort alphabetically for consistent naming
                sorted_types = sorted([v1, v2])
                key = f"{sorted_types[0]}_vs_{sorted_types[1]}"
            
            comparison_data[key] = {'aa': [], 'di': [], 'count': 0}
            all_possible_pairs.append((v1, v2, key))
    
    # Add category for non-target comparisons
    comparison_data['Other_comparisons'] = {'aa': [], 'di': [], 'count': 0}
    
    # Collect data for all sequence pairs
    n_seqs = len(taxa_names)
    total_comparisons = 0
    target_comparisons = 0
    
    for i in range(n_seqs):
        seq1_name = taxa_names[i]
        seq1_type = virus_types[seq1_name]
        
        for j in range(i + 1, n_seqs):
            seq2_name = taxa_names[j]
            seq2_type = virus_types[seq2_name]
            
            aa_dist = aa_matrix[i, j]
            di_dist = di_matrix[i, j]
            total_comparisons += 1
            
            # Check if both are target virus types
            if seq1_type in TARGET_VIRUS_TYPES and seq2_type in TARGET_VIRUS_TYPES:
                # Determine comparison type
                if seq1_type == seq2_type:
                    key = f"{seq1_type}_self"
                else:
                    sorted_types = sorted([seq1_type, seq2_type])
                    key = f"{sorted_types[0]}_vs_{sorted_types[1]}"
                
                comparison_data[key]['aa'].append(aa_dist)
                comparison_data[key]['di'].append(di_dist)
                comparison_data[key]['count'] += 1
                target_comparisons += 1
            else:
                # Non-target virus comparisons
                comparison_data['Other_comparisons']['aa'].append(aa_dist)
                comparison_data['Other_comparisons']['di'].append(di_dist)
                comparison_data['Other_comparisons']['count'] += 1
    
    print(f"  Total processed sequence pairs: {total_comparisons}")
    print(f"  Target virus comparisons: {target_comparisons}")
    
    return comparison_data, all_possible_pairs


def get_comparison_styles():
    """
    Define visualization styles for all comparison types
    
    Returns:
    --------
    dict
        Style dictionary for all possible comparison types
    """
    # Complete predefined style dictionary - all 21 possible combinations
    full_style_dict = {
        # Self-comparisons (6)
        'HRSV_A_self': {'color': '#FF6B6B', 'marker': 'o', 'size': 70, 
                       'label': 'HRSV_A self-comparisons', 'alpha': 0.8},
        'HRSV_B_self': {'color': '#4ECDC4', 'marker': 'o', 'size': 70, 
                       'label': 'HRSV_B self-comparisons', 'alpha': 0.8},
        'BRSV_self': {'color': '#8B4513', 'marker': 'o', 'size': 70, 
                     'label': 'BRSV self-comparisons', 'alpha': 0.8},
        'AMPV_self': {'color': '#06D6A0', 'marker': 'o', 'size': 70, 
                     'label': 'AMPV self-comparisons', 'alpha': 0.8},
        'HMPV_self': {'color': '#FFD166', 'marker': 'o', 'size': 70, 
                     'label': 'HMPV self-comparisons', 'alpha': 0.8},
        'murinePV_self': {'color': '#118AB2', 'marker': 'o', 'size': 70, 
                         'label': 'murinePV self-comparisons', 'alpha': 0.8},
        
        # HRSV_A cross-comparisons (5)
        'HRSV_A_vs_HRSV_B': {'color': '#9C27B0', 'marker': 's', 'size': 85, 
                            'label': 'HRSV_A vs HRSV_B', 'alpha': 0.9},
        'HRSV_A_vs_BRSV': {'color': '#F44336', 'marker': 'D', 'size': 85, 
                          'label': 'HRSV_A vs BRSV', 'alpha': 0.9},
        'HRSV_A_vs_AMPV': {'color': '#E91E63', 'marker': '^', 'size': 85, 
                          'label': 'HRSV_A vs AMPV', 'alpha': 0.9},
        'HRSV_A_vs_HMPV': {'color': '#FF9800', 'marker': 'v', 'size': 85, 
                          'label': 'HRSV_A vs HMPV', 'alpha': 0.9},
        'HRSV_A_vs_murinePV': {'color': '#795548', 'marker': 'p', 'size': 85, 
                              'label': 'HRSV_A vs murinePV', 'alpha': 0.9},
        
        # HRSV_B cross-comparisons (4)
        'HRSV_B_vs_BRSV': {'color': '#2196F3', 'marker': 'D', 'size': 90, 
                          'label': 'HRSV_B vs BRSV', 'alpha': 0.9},
        'HRSV_B_vs_AMPV': {'color': '#03A9F4', 'marker': '^', 'size': 90, 
                          'label': 'HRSV_B vs AMPV', 'alpha': 0.9},
        'HRSV_B_vs_HMPV': {'color': '#00BCD4', 'marker': 'v', 'size': 90, 
                          'label': 'HRSV_B vs HMPV', 'alpha': 0.9},
        'HRSV_B_vs_murinePV': {'color': '#009688', 'marker': 'p', 'size': 90, 
                              'label': 'HRSV_B vs murinePV', 'alpha': 0.9},
        
        # BRSV cross-comparisons (3)
        'BRSV_vs_AMPV': {'color': '#4CAF50', 'marker': '^', 'size': 95, 
                        'label': 'BRSV vs AMPV', 'alpha': 0.9},
        'BRSV_vs_HMPV': {'color': '#8BC34A', 'marker': 'v', 'size': 95, 
                        'label': 'BRSV vs HMPV', 'alpha': 0.9},
        'BRSV_vs_murinePV': {'color': '#CDDC39', 'marker': 'p', 'size': 95, 
                            'label': 'BRSV vs murinePV', 'alpha': 0.9},
        
        # AMPV cross-comparisons (2)
        'AMPV_vs_HMPV': {'color': '#FFC107', 'marker': 'v', 'size': 100, 
                        'label': 'AMPV vs HMPV', 'alpha': 0.9},
        'AMPV_vs_murinePV': {'color': '#FFEB3B', 'marker': 'p', 'size': 100, 
                            'label': 'AMPV vs murinePV', 'alpha': 0.9},
        
        # HMPV cross-comparisons (1)
        'HMPV_vs_murinePV': {'color': '#FF5722', 'marker': 'p', 'size': 105, 
                            'label': 'HMPV vs murinePV', 'alpha': 0.9},
        
        # Other comparisons
        'Other_comparisons': {'color': '#CCCCCC', 'marker': '.', 'size': 25, 
                             'label': 'Other comparisons', 'alpha': 0.2}
    }
    
    return full_style_dict


def create_visualization(comparison_data, comparison_styles):
    """
    Create scatter plot visualization of AA vs 3di distance correlations
    
    Parameters:
    -----------
    comparison_data : dict
        Comparison data organized by virus pair type
    comparison_styles : dict
        Style definitions for each comparison type
    
    Returns:
    --------
    tuple
        (figure, overall_correlation)
    """
    # Configure plot appearance
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    # Collect all data for overall analysis
    all_aa = []
    all_di = []
    
    # First plot other comparisons as background
    if comparison_data['Other_comparisons']['count'] > 0:
        style = comparison_styles.get('Other_comparisons')
        ax.scatter(comparison_data['Other_comparisons']['aa'],
                  comparison_data['Other_comparisons']['di'],
                  alpha=style['alpha'],
                  s=style['size'],
                  color=style['color'],
                  edgecolors='none',
                  label=style['label'],
                  zorder=1)
        
        all_aa.extend(comparison_data['Other_comparisons']['aa'])
        all_di.extend(comparison_data['Other_comparisons']['di'])
    
    # Plot all target virus comparisons that have data
    print("\nPlotting comparisons:")
    for key, data in comparison_data.items():
        if key != 'Other_comparisons' and data['count'] > 0:
            style = comparison_styles.get(key)
            if style:
                ax.scatter(data['aa'],
                          data['di'],
                          alpha=style['alpha'],
                          s=style['size'],
                          color=style['color'],
                          edgecolors='black' if '_self' not in key else style['color'],
                          linewidth=1.0,
                          marker=style['marker'],
                          label=style['label'],
                          zorder=2)
                
                all_aa.extend(data['aa'])
                all_di.extend(data['di'])
                print(f"  Plotted: {key} with {data['count']} points")
    
    # Convert to numpy arrays for statistical analysis
    all_aa = np.array(all_aa)
    all_di = np.array(all_di)
    
    overall_corr = None
    
    if len(all_aa) > 1:
        # Calculate overall correlation
        overall_corr, overall_p = stats.pearsonr(all_aa, all_di)
        
        # Add regression line
        z = np.polyfit(all_aa, all_di, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, max(all_aa), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'k-', linewidth=3.5, alpha=0.8,
                label=f'Overall regression line (r={overall_corr:.3f})',
                zorder=3)
    
    # Add diagonal reference line (y = x)
    max_val = max(max(all_aa), max(all_di)) if len(all_aa) > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], 'b--', alpha=0.6, linewidth=2.5,
            label='y = x reference line', zorder=3)
    
    # Set figure properties
    ax.set_xlabel('AA Sequence Distance (p-distance)', 
                 fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_ylabel('3di Structural Distance (p-distance)', 
                 fontsize=FONT_SIZES['label'], fontweight='bold')
    ax.set_title('Correlation Analysis: AA Sequence Distance vs 3di Structural Distance\n'
                'All Pairwise Comparisons among Six Virus Types', 
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=30)
    
    # Set axis properties
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.set_xlim(-0.02, max_val * 1.05)
    ax.set_ylim(-0.02, max_val * 1.05)
    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
    
    # Add legend
    ax.legend(fontsize=FONT_SIZES['legend'], loc='upper left', 
              bbox_to_anchor=(1.02, 1), borderaxespad=0., 
              framealpha=0.9, fancybox=True, ncol=1)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.68, 1])
    
    return fig, overall_corr


def print_statistics(virus_counts, comparison_data, overall_corr):
    """
    Print analysis statistics to console
    
    Parameters:
    -----------
    virus_counts : dict
        Count of sequences per virus type
    comparison_data : dict
        Comparison data with counts
    overall_corr : float
        Overall correlation coefficient
    """
    print("\n" + "=" * 80)
    print("ANALYSIS STATISTICS")
    print("=" * 80)
    
    print("\nSequence classification:")
    for vtype, count in virus_counts.items():
        if count > 0:
            print(f"  {vtype}: {count} sequences")
    
    print("\nTarget virus type detection:")
    for vtype in TARGET_VIRUS_TYPES:
        count = virus_counts.get(vtype, 0)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {vtype}: {count} sequences")
    
    print("\nComparison data statistics:")
    for key, data in comparison_data.items():
        if data['count'] > 0:
            if key == 'Other_comparisons':
                print(f"  Other comparisons: {data['count']} comparisons")
            else:
                # Parse key for display
                if '_self' in key:
                    vtype = key.replace('_self', '')
                    display_name = f"{vtype} self-comparisons"
                else:
                    parts = key.split('_vs_')
                    display_name = f"{parts[0]} vs {parts[1]}"
                print(f"  {display_name}: {data['count']} comparisons")
    
    if overall_corr is not None:
        print(f"\nOverall correlation coefficient: r = {overall_corr:.4f}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to execute virus distance correlation analysis
    """
    print("=" * 80)
    print("Virus Sequence-Structure Distance Correlation Analysis")
    print("=" * 80)
    
    # Construct full file paths
    aa_full_path = os.path.join(BASE_DIR, AA_FILE)
    di_full_path = os.path.join(BASE_DIR, DI_FILE)
    
    # Check if files exist
    for file_path, file_desc in [(aa_full_path, "Amino acid alignment"),
                                 (di_full_path, "3di structural alignment")]:
        if not os.path.exists(file_path):
            print(f"ERROR: {file_desc} file not found - {file_path}")
            print("Please update the file paths in the script configuration.")
            return
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Amino acid alignment: {AA_FILE}")
    print(f"3di structural alignment: {DI_FILE}")
    print("-" * 80)
    
    try:
        # Step 1: Load sequence files
        aa_alignment, di_alignment, success = load_sequence_files(BASE_DIR, AA_FILE, DI_FILE)
        if not success:
            return
        
        # Step 2: Get sequence names
        taxa_names = [record.id for record in aa_alignment]
        
        # Step 3: Identify virus types
        virus_types, virus_counts = identify_virus_types(taxa_names)
        
        # Step 4: Calculate distance matrices
        aa_matrix, di_matrix = calculate_distance_matrices(aa_alignment, di_alignment)
        
        # Step 5: Collect comparison data
        comparison_data, all_possible_pairs = collect_comparison_data(
            taxa_names, virus_types, aa_matrix, di_matrix
        )
        
        # Step 6: Get visualization styles
        comparison_styles = get_comparison_styles()
        
        # Step 7: Create visualization
        fig, overall_corr = create_visualization(comparison_data, comparison_styles)
        
        # Step 8: Save figure
        output_file = 'all_virus_pairwise_comparisons_complete2.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nFigure saved as: {output_file}")
        
        # Step 9: Print statistics
        print_statistics(virus_counts, comparison_data, overall_corr)
        
        # Return results for potential further analysis
        return {
            'taxa_names': taxa_names,
            'virus_types': virus_types,
            'virus_counts': virus_counts,
            'comparison_data': comparison_data,
            'overall_correlation': overall_corr
        }
        
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Unexpected error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = main()