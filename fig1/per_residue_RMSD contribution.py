#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RMSD Analysis Script
================================================================================

Description:
This script calculates per-residue RMSD contributions from FoldSeek structural
alignments of HRSV (Human Respiratory Syncytial Virus) F protein variants.

Functionality:
1. Loads PDB structures and FoldSeck alignment results
2. Performs structural superposition of aligned residues
3. Calculates RMSD for each aligned residue pair
4. Outputs residue-level RMSD contributions for downstream analysis

Input Requirements:
1. PDB structure files for each query/target ID
2. FoldSeek alignment results in TSV format

Output:
- rmsd_contributions_merged.csv: CSV file containing per-residue RMSD values
  Columns: query, target, residue_number, rmsd_contribution, total_rmsd, aligned_length

Usage:
1. Install dependencies: pip install pandas numpy biopython
2. Configure paths below (PDB_DIR, ALN_RESULTS, OUTPUT_DIR)
3. Run: python this_script.py

Dependencies:
- pandas>=1.3.0
- numpy>=1.21.0
- biopython>=1.79

Author: [Your Name/Affiliation]
Contact: [Your Email]
Date: [Date]
================================================================================
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - USER MUST MODIFY THESE PATHS
# ============================================================================

# Path to directory containing PDB files (must contain .pdb files)
PDB_DIR = "/path/to/your/pdb/files/"

# Path to FoldSeek alignment results (TSV format)
ALN_RESULTS = "/path/to/your/foldseek/results.tsv"

# Output directory for results
OUTPUT_DIR = "/path/to/output/directory/"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names for FoldSeek TSV output
COLUMN_NAMES = ['query', 'target', 'qstart', 'qend', 'qaln', 'taln', 'evalue', 'rmsd']

# ============================================================================
# PDB STRUCTURE LOADING FUNCTION
# ============================================================================

def load_pdb_structure(pdb_id, pdb_dir):
    """
    Load PDB structure and extract C-alpha atoms
    
    Parameters:
    -----------
    pdb_id : str
        PDB identifier (without .pdb extension)
    pdb_dir : str
        Directory containing PDB files
    
    Returns:
    --------
    list or None
        List of C-alpha atoms if successful, None otherwise
    """
    try:
        parser = PDBParser(QUIET=True)
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        
        if os.path.exists(pdb_path):
            structure = parser.get_structure(pdb_id, pdb_path)
            ca_atoms = []
            
            # Extract C-alpha atoms from all chains
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atoms.append(residue['CA'])
            
            return ca_atoms
        else:
            print(f"Warning: PDB file not found for {pdb_id} at {pdb_path}")
            return None
            
    except Exception as e:
        print(f"Error loading PDB {pdb_id}: {str(e)}")
        return None

# ============================================================================
# ALIGNMENT PROCESSING FUNCTION
# ============================================================================

def process_alignment(row):
    """
    Process a single structural alignment and calculate RMSD contributions
    
    Parameters:
    -----------
    row : pd.Series
        Row from alignment DataFrame containing alignment information
    
    Returns:
    --------
    dict or None
        Dictionary with alignment results if successful, None otherwise
    """
    try:
        query_id = row['query']
        target_id = row['target']
        
        print(f"Processing alignment: {query_id} vs {target_id}")
        
        # Skip self-alignments
        if query_id == target_id:
            return None
        
        # Load structures
        q_ca = load_pdb_structure(query_id, PDB_DIR)
        t_ca = load_pdb_structure(target_id, PDB_DIR)
        
        if q_ca is None or t_ca is None:
            return None
        
        # Get alignment sequences
        qaln, taln = row['qaln'], row['taln']
        
        # Extract aligned C-alpha atoms
        q_atoms = []
        t_atoms = []
        q_pos = t_pos = 0
        
        for qa, ta in zip(qaln, taln):
            if qa != '-' and ta != '-':
                if q_pos < len(q_ca) and t_pos < len(t_ca):
                    q_atoms.append(q_ca[q_pos])
                    t_atoms.append(t_ca[t_pos])
                q_pos += 1
                t_pos += 1
            elif qa != '-':
                q_pos += 1
            elif ta != '-':
                t_pos += 1
        
        # Need at least 3 atoms for RMSD calculation
        if len(q_atoms) < 3:
            print(f"Warning: Insufficient aligned atoms ({len(q_atoms)}) for {query_id} vs {target_id}")
            return None
        
        # Calculate overall RMSD
        sup = Superimposer()
        sup.set_atoms(q_atoms, t_atoms)
        rmsd = sup.rms
        
        # Calculate per-residue RMSD contributions
        residue_contrib = {}
        atom_index = 0
        q_pos = 0
        
        for qa, ta in zip(qaln, taln):
            if qa != '-' and ta != '-':
                if q_pos < len(q_ca) and atom_index < len(q_atoms):
                    residue = q_ca[q_pos].get_parent()
                    res_num = residue.id[1]  # Residue number
                    
                    # Calculate squared distance for this residue
                    diff = q_atoms[atom_index].get_coord() - t_atoms[atom_index].get_coord()
                    dist_sq = np.sum(diff**2)
                    residue_contrib[res_num] = dist_sq
                    
                    atom_index += 1
                q_pos += 1
            elif qa != '-':
                q_pos += 1
        
        if not residue_contrib:
            return None
            
        # Normalize per-residue contributions to match total RMSD
        total_dist_sq = sum(residue_contrib.values())
        scale_factor = (rmsd**2 * len(residue_contrib)) / total_dist_sq if total_dist_sq > 0 else 0
        
        # Convert squared distances to RMSD values
        for res_num in residue_contrib:
            residue_contrib[res_num] = np.sqrt(residue_contrib[res_num] * scale_factor)
        
        # Return alignment results
        return {
            'query': query_id,
            'target': target_id,
            'total_rmsd': rmsd,
            'residue_contributions': residue_contrib,
            'aligned_length': len(q_atoms)
        }
        
    except Exception as e:
        print(f"Error processing alignment {row['query']} vs {row['target']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# RESULTS WRITING FUNCTION
# ============================================================================

def write_results_to_file(results, output_file, write_header=False):
    """
    Write alignment results to CSV file
    
    Parameters:
    -----------
    results : list
        List of alignment result dictionaries
    output_file : str
        Path to output CSV file
    write_header : bool
        Whether to write column headers
    """
    output_data = []
    for res in results:
        for res_num, val in res['residue_contributions'].items():
            output_data.append({
                'query': res['query'],
                'target': res['target'],
                'residue_number': res_num,
                'rmsd_contribution': val,
                'total_rmsd': res['total_rmsd'],
                'aligned_length': res['aligned_length']
            })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, mode='a', header=write_header, index=False)
    
    # Clean up to save memory
    del output_data, output_df

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to process all alignments
    """
    print("Starting HRSV RMSD analysis...")
    print("=" * 60)
    
    # Check if input files/directories exist
    if not os.path.exists(PDB_DIR):
        print(f"ERROR: PDB directory not found: {PDB_DIR}")
        print("Please update the PDB_DIR path in the script configuration.")
        return
    
    if not os.path.exists(ALN_RESULTS):
        print(f"ERROR: Alignment results file not found: {ALN_RESULTS}")
        print("Please update the ALN_RESULTS path in the script configuration.")
        return
    
    print(f"PDB directory: {PDB_DIR}")
    print(f"Alignment file: {ALN_RESULTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    try:
        # Load alignment results
        print("Loading alignment results...")
        df = pd.read_csv(ALN_RESULTS, sep='\t', header=None, names=COLUMN_NAMES)
        print(f"Loaded {len(df)} alignments")
        
        # Set output file path
        output_file = os.path.join(OUTPUT_DIR, "rmsd_contributions_merged.csv")
        
        # Batch processing parameters (for memory management)
        batch_size = 1000  # Process 1000 alignments per batch
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        print(f"Processing in {total_batches} batches...")
        
        # Process each batch
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            
            print(f"\nBatch {batch_num + 1}/{total_batches} (alignments {start_idx}-{end_idx-1})")
            
            batch_results = []
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                result = process_alignment(row)
                if result:
                    batch_results.append(result)
            
            # Write batch results to file
            write_header = (batch_num == 0)
            write_results_to_file(batch_results, output_file, write_header)
            
            # Clean up memory
            del batch_results
            
            print(f"Completed batch {batch_num + 1}/{total_batches}")
        
        print("\n" + "=" * 60)
        print(f"Analysis complete!")
        print(f"Results saved to: {output_file}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check the file paths and permissions.")
    except pd.errors.EmptyDataError:
        print("ERROR: Alignment results file is empty or incorrectly formatted.")
    except Exception as e:
        print(f"Unexpected error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()