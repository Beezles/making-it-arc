from functions import *
import os
import sys
import urllib.request
import shutil
from biopandas.pdb import PandasPdb
import shlex
from tqdm import tqdm
import time
import numpy as np
import torch
import mdtraj as md
from six import PY2
from Bio.PDB import is_aa
from Bio.PDB import PDBParser, PDBIO, Select
import inquirer
import MDAnalysis as mda
from MDAnalysis.analysis import align
from functions import _plot_rmsd
from functions import _plot_cross_correlation
import pytraj as pt
import pandas as pd
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *

def main():
    # Print program introduction
    print_intro()
    
    # Check GPU compatibility
    gpu_check()
    
    # Setup working environment
    workDir = setup_environment()
    
    # Get PDB file (either local or download from RCSB)
    pdb_file = get_pdb_file(workDir)
    
    # Process PDB file
    processed_pdb, secondary_structure = process_pdb(pdb_file, workDir)
    
    # Generate Martini topology
    martini_topology(workDir, processed_pdb, secondary_structure)
    
    # Get MD parameters and setup simulation
    params = get_md_parameters(workDir)
    simulation, gro, top = setup_simulation(params)
    
    # Run equilibration
    run_equilibration(params, simulation, gro)
    
    # Get production parameters and run production MD
    prod_params = get_production_parameters(workDir)
    run_equilibration(prod_params, workDir)
    
    # Process trajectory and backmap if requested
    process_trajectory(prod_params, workDir)
    
    # Offer analysis options
    offer_analysis_options(workDir, prod_params)

    # Load trajectories for analysis
    nw_dcd = os.path.join(workDir, f"{params['jobname']}_cg_nw.dcd")
    nw_pdb = os.path.join(workDir, f"{params['jobname']}_nw.pdb")
    traj_atomistic = os.path.join(workDir, f"{params['jobname']}_backmapped_align.dcd")
    pdb_ref_atomistic = os.path.join(workDir, f"{params['jobname']}_backmapped_align.pdb")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [nw_dcd, nw_pdb, traj_atomistic, pdb_ref_atomistic]):
        print("\n> Error: Required trajectory files not found for analysis")
        return
    
    print("\n> Loading trajectories for analysis...")
    traj_load_cg = pt.load(nw_dcd, nw_pdb)
    traj_load_atomistic = pt.load(traj_atomistic, pdb_ref_atomistic)
    
    # Run selected analyses
    if 'rmsd' in inquirer.answers['analysis_options']:
        print("\n> Calculating RMSD...")
        RMSD_compute(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    if 'gyration' in inquirer.answers['analysis_options']:
        print("\n> Calculating Radius of Gyration...")
        gy_radius(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    if 'rmsf' in inquirer.answers['analysis_options']:
        print("\n> Calculating RMSF...")
        RMSF_graph(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    if '2d_rmsd' in inquirer.answers['analysis_options']:
        print("\n> Calculating 2D RMSD...")
        _plot_rmsd(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    if 'pca' in inquirer.answers['analysis_options']:
        print("\n> Performing PCA Analysis...")
        _plot_cross_correlation(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    if 'cross_corr' in inquirer.answers['analysis_options']:
        print("\n> Calculating Cross Correlation...")
        pearson_CC(workDir, params['jobname'], traj_load_cg, traj_load_atomistic)
    
    print("\n> All selected analyses completed!")

if __name__ == "__main__":
    main()