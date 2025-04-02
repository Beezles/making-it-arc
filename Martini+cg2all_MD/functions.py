
import os
import sys
import shutil
import urllib.request
import subprocess
import numpy as np
import pandas as pd
import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry.hbond import _prep_kabsch_sander_arrays
from mdtraj.geometry import _geometry
import matplotlib.pyplot as plt
from tqdm import tqdm
import inquirer
import torch
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import pytraj as pt
import parmed
import shlex, subprocess
from Bio.PDB import is_aa, PDBParser, PDBIO, Select
from biopandas.pdb import PandasPdb
from openmm import *
from openmm.app import *
from openmm.unit import *
import martini_openmm as martini
from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from openmm.unit import (
    femtosecond, picosecond, nanometer, kelvin, bar, 
    kilojoule_per_mole, kilocalorie_per_mole
)
from sys import stdout
from MDAnalysis.analysis import align, rms, diffusion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytraj import matrix

def print_intro():
    ###Print program introduction###
    print("\n" + "="*80)
    print("Martini+cg2all MD Simulation Program".center(80))
    print("="*80)
    print("This is a molecular dynamics simulation program using Martini+cg2all")
    print("Modeled after making-it-rain's Martini+cg2all Google Colab version")
    print("For original documentation visit:")
    print("https://github.com/pablo-arantes/making-it-rain")
    print("="*80 + "\n")

def _create_water_free_trajectory(universe, output_dcd, output_pdb, skip=1):
    """Create a water-free trajectory"""
    nw_mask = "not (resname W or name NA+ or name CL-)"
    nw = universe.select_atoms(nw_mask)
    with mda.Writer(output_dcd, nw.n_atoms) as W:
        for ts in universe.trajectory[::skip]:
            W.write(nw)
    nw.write(output_pdb)

def _create_full_trajectory(universe, output_dcd, output_pdb, skip=1):
    """Create full system trajectory"""
    with mda.Writer(output_dcd, universe.atoms.n_atoms) as W:
        for ts in universe.trajectory[::skip]:
            W.write(universe.atoms)
    universe.atoms.write(output_pdb)
    
def offer_analysis_options(workDir, params):
    ###Offer various analysis options to the user###
    questions = [
        inquirer.Confirm('run_analysis',
                        message="Would you like to perform analysis on the trajectory?",
                        default=True),
        inquirer.Checkbox('analysis_options',
                        message="Select analyses to run (space to select, enter to confirm)",
                        choices=[
                            ('RMSD Calculation', 'rmsd'),
                            ('Radius of Gyration', 'gyration'),
                            ('RMSF Analysis', 'rmsf'),
                            ('2D RMSD', '2d_rmsd'),
                            ('PCA Analysis', 'pca'),
                            ('Cross Correlation', 'cross_corr')
                        ],
                        default=['rmsd', 'gyration'])
    ]
    answers = inquirer.prompt(questions)
    
    if not answers['run_analysis']:
        print("\n> Analysis skipped. Simulation complete!")
        return
    
def copy_with_progress(src, dst):
    ###Copy files with progress bar###
    with tqdm(total=os.path.getsize(src), unit='B', unit_scale=True, 
             desc=f"Copying {os.path.basename(src)}") as pbar:
        shutil.copy2(src, dst, follow_symlinks=True)
        pbar.update(os.path.getsize(src))

def gpu_check():
    ###Check GPU availability and compatibility###
    if not torch.cuda.is_available():
        print('GPU not available')
        return False
    
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv'])
    gpu_info = gpu_info.decode('utf-8').strip().split('\n')[1]
    print(f"GPU detected: {gpu_info}")
    return True

def select_file(directory="."):
    ###Interactive file selection using inquirer###
    while True:
        try:
            files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
            if not files:
                raise FileNotFoundError("No PDB files found in directory")
            
            questions = [
                inquirer.List('pdb_file',
                            message="Select PDB file",
                            choices=files,
                            default=files[0])
            ]
            answers = inquirer.prompt(questions)
            return os.path.join(directory, answers['pdb_file'])
        except Exception as e:
            print(f"Error: {e}")
            if not inquirer.confirm("Try again?", default=True):
                sys.exit(1)

def setup_environment():
    ###Configure working environment interactively###
    questions = [
        inquirer.Text('work_dir',
                    message="Working directory path",
                    default=os.path.join(os.getcwd(), "Proteins")),
        inquirer.Confirm('download_martini',
                        message="Copy martini forcefield files?",
                        default=True)
    ]
    answers = inquirer.prompt(questions)
    
    os.makedirs(answers['work_dir'], exist_ok=True)
    
    if answers['download_martini']:
        martini_dir = os.path.join(answers['work_dir'], "martini")
        if not os.path.exists(martini_dir) and os.path.exists("./martini"):
            print("Copying martini files...")
            os.makedirs(martini_dir, exist_ok=True)
            for item in os.listdir("./martini"):
                src = os.path.join("./martini", item)
                dst = os.path.join(martini_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    
    return answers['work_dir']

def get_pdb_file(work_dir):
    ###Handle PDB file selection/download###
    questions = [
        inquirer.List('pdb_source',
                    message="PDB file source",
                    choices=[
                        ('Use local file', 'local'),
                        ('Download from RCSB', 'rcsb')
                    ],
                    default='local')
    ]
    answers = inquirer.prompt(questions)
    
    if answers['pdb_source'] == 'local':
        return select_file(work_dir)
    else:
        pdb_id = inquirer.text("Enter PDB ID (e.g. 1ABC):").execute()
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        out_path = os.path.join(work_dir, f"{pdb_id}.pdb")
        
        print(f"Downloading {pdb_id}...")
        try:
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)

def process_pdb(pdb_file, work_dir):
    ###Process PDB file with interactive options###
    questions = [
        inquirer.Confirm('remove_waters',
                       message="Remove water molecules?",
                       default=True),
        inquirer.Confirm('remove_hydrogens',
                       message="Remove hydrogen atoms?",
                       default=True),
        inquirer.Confirm('standard_aa_only',
                       message="Keep only standard amino acids?",
                       default=True)
    ]
    answers = inquirer.prompt(questions)
    
    starting = os.path.join(work_dir, "starting.pdb")
    starting2 = os.path.join(work_dir, "starting2.pdb")
    starting_end = os.path.join(work_dir, "starting_end.pdb")

    print("\nProcessing PDB file...")
    with tqdm(total=4, desc="Overall progress") as pbar:
        # Step 1: Initial processing
        ppdb = PandasPdb().read_pdb(pdb_file)
        if answers['remove_waters']:
            ppdb.df['HETATM'] = ppdb.df['HETATM'][ppdb.df['HETATM']['residue_name'] == 'HOH']
        ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] != 'OXT']
        if answers['remove_hydrogens']:
            ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
        ppdb.to_pdb(path=starting, records=['ATOM', 'HETATM'], gz=False, append_newline=True)
        pbar.update(1)
        
        # Step 2: Select standard amino acids
        if answers['standard_aa_only']:
            class ProtSelect(Select):
                def accept_residue(self, residue):
                    return is_aa(residue, standard=True)
            
            pdb_ini = PDBParser().get_structure("pdb", starting)
            io = PDBIO()
            io.set_structure(pdb_ini)
            io.save(starting2, ProtSelect())
        else:
            shutil.copy(starting, starting2)
        pbar.update(1)
        
        # Step 3: Run pdb4amber
        cmd = f"pdb4amber -i {starting2} -o {starting_end} -p"
        try:
            subprocess.run(cmd, shell=True, check=True)
            pbar.update(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running pdb4amber: {e}")
            sys.exit(1)
        
        # Step 4: Calculate secondary structure
        pdb_ss = md.load_pdb(starting2)
        ss = compute_dssp(pdb_ss, simplified=True)
        my_string = ''.join([str(item) for sublist in ss for item in sublist])
        pbar.update(1)
    
    print(f"\nSecondary structure: {my_string}")
    return starting_end, my_string

def compute_dssp(traj, simplified=True):
    ###Compute Dictionary of protein secondary structure assignments###
    if traj.topology is None:
        raise ValueError('DSSP calculation requires topology')

    SIMPLIFIED_CODE_TRANSLATION = str.maketrans('HGIEBTS ', 'HHHEECCC')
    
    xyz, nco_indices, ca_indices, proline_indices, protein_indices \
        = _prep_kabsch_sander_arrays(traj)
    chain_ids = np.array([r.chain.index for r in traj.top.residues], dtype=np.int32)

    value = _geometry._dssp(xyz, nco_indices, ca_indices, proline_indices, chain_ids)
    if simplified:
        value = value.translate(SIMPLIFIED_CODE_TRANSLATION)

    array = np.fromiter(value, dtype='U2').reshape(len(xyz), len(nco_indices))
    array[:, np.logical_not(protein_indices)] = 'NA'
    return array

def _prep_kabsch_sander_arrays(traj):
    ###Helper function for DSSP calculation###
    # Implementation would go here
    pass

def martini_topology(workDir: str, starting2: str, my_string: str) -> Tuple[str, str, str]:
    ###Generate Martini topology with interactive parameter selection###
    # Initialize file paths
    workDir = Path(workDir)
    top_martini = workDir / "martini.top"
    top_temp = workDir / "martini_temp.top"
    gro_martini = workDir / "system.gro"
    pdb_nw_martini = workDir / "martini.pdb"
    pdb_martini = workDir / "ions.pdb"

    questions = [
        # [Previous questions list remains the same...]
    ]
    
    answers = inquirer.prompt(questions)
    
    # Process answers
    ss_h = my_string if answers['secondary_structure'] else ''
    elastic = (f" -elastic -ef {answers['elastic_force']} -el {answers['elastic_lower']} "
              f"-eu {answers['elastic_upper']}") if answers['elastic_bonds'] else ''
    scfix = " -scfix" if answers['side_chain_corrections'] else ''
    cys = "auto" if answers['cystein_bonds'] else "none"

    # Clean up existing files
    for f in [top_martini, gro_martini, pdb_nw_martini]:
        if f.exists():
            f.unlink()

    # Generate martinize2 command
    martinize_cmd = (
        f"martinize2 -f {starting2} -x {pdb_nw_martini} -ss {ss_h} "
        f"-ff {answers['force_field']}{elastic} -p {answers['position_restraints']} "
        f"-pf {answers['posres_force']} -cys {cys}{scfix} -o {top_martini} "
        "-maxwarn 10 -ignh"
    )

    # Write and execute script
    with open(workDir / 'martinize.sh', 'w') as f:
        f.write(f"cd {workDir}\n{martinize_cmd}\n")
    
    (workDir / 'martinize.sh').chmod(0o700)
    subprocess.run(f"bash {workDir/'martinize.sh'}", shell=True, check=True)

    # Solvate system
    insane_cmd = (
        f"insane -o {gro_martini} -p {top_temp} -f {pdb_nw_martini} "
        f"-d {answers['size_box']} -sol W -salt {answers['concentration']} "
        f"-charge auto -pbc {answers['box_type']}"
    )
    subprocess.run(insane_cmd, shell=True, check=True)

    # Update topology file
    with open(top_martini, 'r') as f:
        lines = f.readlines()[2:]  # Skip first two lines

    ff_includes = {
        "martini3001": [
            '#include "martini/martini_v3.0.0.itp"\n',
            '#include "martini/martini_v3.0.0_ions.itp"\n',
            '#include "martini/martini_v3.0.0_solvents.itp"\n'
        ],
        "martini22": [
            '#include "martini/martini_v2.2.itp"\n',
            '#include "martini/martini_v2.2_ions.itp"\n',
            '#include "martini/martini_v2.2_aminoacids.itp"\n'
        ]
    }
    
    with open(top_martini, 'w') as f:
        f.writelines(ff_includes[answers['force_field']] + lines)

    # Handle concentration-dependent topology updates
    with open(top_temp) as f_in, open(top_martini, 'a') as f_out:
        lines = f_in.readlines()
        num_lines = 1 if float(answers['concentration']) == 0 else 3
        f_out.writelines(lines[-num_lines:])

    # Write final PDB
    universe = mda.Universe(str(gro_martini))
    with mda.Writer(str(pdb_martini)) as pdb:
        pdb.write(universe)

    # Cleanup
    for temp_file in workDir.glob('*#'):
        temp_file.unlink()
    top_temp.unlink()

    # Verify success
    if not (gro_martini.exists() and top_martini.exists()):
        print("\nERROR: Check your input files!")
        sys.exit(1)
    
    print("\nSuccessfully generated topology!")
    return str(gro_martini), str(top_martini), str(pdb_martini)


def _prep_kabsch_sander_arrays(traj: md.Trajectory) -> Tuple:
    ###Prepare arrays for Kabsch-Sander DSSP calculation###
    # Implementation would use mdtraj internals
    pass

@dataclass
class MDParameters:
    jobname: str
    files: Dict[str, Path]
    minimization_steps: int
    timestep: int
    nvt_time: float
    npt_time: float
    temperature: float
    pressure: float
    traj_freq: int
    log_freq: int

def get_md_parameters(work_dir: Union[str, Path]) -> MDParameters:
    ###Interactive parameter selection for MD equilibration###
    work_dir = Path(work_dir)
    questions = [
        inquirer.Text('jobname',
                     message="Job name",
                     default='T2_1000',
                     validate=lambda _, x: x.strip() != ''),
        # ... other questions remain the same ...
    ]
    
    answers = inquirer.prompt(questions)
    
    return MDParameters(
        jobname=str(work_dir / answers['jobname']),
        files={
            'coordinate': work_dir / "system.gro",
            'pdb': work_dir / "ions.pdb",
            'topology': work_dir / "martini.top"
        },
        minimization_steps=int(answers['minimization_steps']),
        timestep=int(answers['timestep']),
        nvt_time=float(answers['nvt_time']),
        npt_time=float(answers['npt_time']),
        temperature=float(answers['temperature']),
        pressure=float(answers['pressure']),
        traj_freq=int(answers['traj_freq']),
        log_freq=int(answers['log_freq'])
    )

def setup_simulation(params: MDParameters) -> Tuple[Simulation, GromacsGroFile, martini.MartiniTopFile]:
    ###Configure the OpenMM simulation system###
    print("\n> Setting up the system:")
    
    # Convert parameters to proper units
    dt = params.timestep * femtosecond
    temperature = params.temperature * kelvin
    pressure = params.pressure * bar
    
    with tqdm(total=4, desc="Loading files") as pbar:
        gro = GromacsGroFile(str(params.files['coordinate']))
        pbar.update(1)
        
        defines = _load_defines()
        top = martini.MartiniTopFile(
            str(params.files['topology']),
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            defines=defines
        )
        pbar.update(2)
        
        system = top.create_system(nonbonded_cutoff=1.1*nanometer)
        integrator = LangevinIntegrator(temperature, 10.0/picosecond, dt)
        integrator.setRandomNumberSeed(0)
        simulation = Simulation(top.topology, system, integrator)
        simulation.context.setPositions(gro.positions)
        pbar.update(1)
    
    return simulation, gro, top

def _load_defines() -> Dict[str, bool]:
    ###Load defines from file if exists###
    defines = {}
    try:
        with open("defines.txt") as def_file:
            defines = {line.strip(): True for line in def_file}
    except FileNotFoundError:
        pass
    return defines

def run_equilibration(params: MDParameters, 
                    simulation: Simulation, 
                    gro: GromacsGroFile) -> None:
    ###Run NVT and NPT equilibration with progress tracking###
    # Convert parameters
    dt = params.timestep * femtosecond
    temperature = params.temperature * kelvin
    pressure = params.pressure * bar
    
    # NVT Equilibration
    _run_nvt_equilibration(params, simulation, dt, temperature)
    
    # NPT Equilibration
    _run_npt_equilibration(params, simulation, dt, temperature, pressure)

def _run_nvt_equilibration(params: MDParameters,
                          simulation: Simulation,
                          dt: Quantity,
                          temperature: Quantity) -> None:
    ###Run NVT equilibration phase###
    print("\n> Running NVT Equilibration:")
    time_ps = params.nvt_time * 1000
    simulation_time = time_ps * picosecond
    nsteps = int(simulation_time / dt)
    
    # Minimization
    with tqdm(total=params.minimization_steps, desc="Energy minimization") as pbar:
        simulation.minimizeEnergy(
            tolerance=1.0,
            maxIterations=params.minimization_steps,
            callback=lambda step, _: pbar.update(step - pbar.n)
        )
    
    # Setup reporters
    dcd_file = f"{params.jobname}_eq_nvt.dcd"
    simulation.reporters.extend([
        DCDReporter(dcd_file, params.traj_freq),
        StateDataReporter(stdout, params.log_freq, step=True, speed=True,
                         progress=True, totalSteps=nsteps,
                         remainingTime=True, separator='\t\t'),
        StateDataReporter(f"{params.jobname}_eq_nvt.log", params.log_freq, step=True,
                         kineticEnergy=True, potentialEnergy=True,
                         totalEnergy=True, temperature=True)
    ])
    
    # Run NVT
    simulation.context.setVelocitiesToTemperature(temperature)
    _run_simulation_steps(simulation, nsteps, "NVT equilibration")

def _run_npt_equilibration(params: MDParameters,
                          simulation: Simulation,
                          dt: Quantity,
                          temperature: Quantity,
                          pressure: Quantity) -> None:
    ###Run NPT equilibration phase###
    print("\n> Running NPT Equilibration:")
    time_ps = params.npt_time * 1000
    simulation_time = time_ps * picosecond
    nsteps = int(simulation_time / dt)
    
    # Add barostat
    system = simulation.system
    system.addForce(MonteCarloBarostat(pressure, temperature))
    
    # Setup reporters
    simulation.reporters.clear()
    simulation.reporters.extend([
        DCDReporter(f"{params.jobname}_eq_npt.dcd", params.traj_freq),
        StateDataReporter(stdout, params.log_freq, step=True, speed=True,
                         progress=True, totalSteps=nsteps,
                         remainingTime=True, separator='\t\t'),
        StateDataReporter(f"{params.jobname}_eq_npt.log", params.log_freq, step=True,
                         kineticEnergy=True, potentialEnergy=True,
                         totalEnergy=True, temperature=True,
                         volume=True, speed=True)
    ])
    
    # Run NPT
    _run_simulation_steps(simulation, nsteps, "NPT equilibration")

def _run_simulation_steps(simulation: Simulation,
                         nsteps: int,
                         description: str) -> None:
    ###Common function to run simulation steps with progress###
    with tqdm(total=nsteps, desc=description) as pbar:
        simulation.step(nsteps, callback=lambda step, _: pbar.update(step - pbar.n))

def get_production_parameters(work_dir: Union[str, Path]) -> Dict[str, Any]:
    ###Interactive parameter selection for MD production###
    work_dir = Path(work_dir)
    questions = [
        inquirer.Text('jobname',
                     message="Job name (same as equilibration)",
                     default='T2_1000',
                     validate=lambda _, x: x.strip() != ''),
        # ... other questions remain the same ...
    ]
    
    answers = inquirer.prompt(questions)
    
    return {
        'jobname': str(work_dir / answers['jobname']),
        'files': {
            'equil_pdb': work_dir / f"{answers['jobname']}_eq_npt.pdb",
            'state_file': work_dir / f"{answers['jobname']}_eq_npt.rst"
        },
        'stride_time': float(answers['stride_time']),
        'n_strides': int(answers['n_strides']),
        'timestep': int(answers['timestep']),
        'temperature': float(answers['temperature']),
        'pressure': float(answers['pressure']),
        'traj_freq': int(answers['traj_freq']),
        'log_freq': int(answers['log_freq']),
        'continue': answers['continue_sim']
    }

# [Continuing with other functions...]

def process_trajectory(params: Dict[str, Any], work_dir: Union[str, Path]) -> None:
    ###Process and analyze trajectory###
    work_dir = Path(work_dir)
    questions = [
        inquirer.List('skip_frames',
                     message="Skip every N frames for analysis",
                     choices=[1, 2, 5, 10, 20, 50],
                     default=1),
        # ... other questions remain the same ...
    ]
    answers = inquirer.prompt(questions)
    
    template = work_dir / f"{params['jobname']}_prod_%s.dcd"
    pdb_ref = work_dir / f"{params['jobname']}_eq_npt.pdb"
    
    # Load and process trajectory
    flist = [template % i for i in range(1, params['n_strides'] + 1)]
    _process_trajectory_files(flist, pdb_ref, answers, work_dir, params['jobname'])

def _process_trajectory_files(flist: List[Path],
                            pdb_ref: Path,
                            answers: Dict[str, Any],
                            work_dir: Path,
                            jobname: str) -> None:
    ###Helper function to process trajectory files###
    with tqdm(total=len(flist) + 2, desc="Processing trajectory") as pbar:
        u1 = mda.Universe(str(pdb_ref), [str(f) for f in flist])
        u2 = mda.Universe(str(pdb_ref), str(pdb_ref))
        u2.trajectory[0]
        
        align.AlignTraj(u1, u2, select='name CA', in_memory=True).run()
        pbar.update(1)
        
        if answers['remove_waters']:
            _create_water_free_trajectory(u1, u2, work_dir, jobname, answers['skip_frames'])
            pbar.update(1)
        
        _create_full_trajectory(u1, u2, work_dir, jobname, answers['skip_frames'])
        pbar.update(1)
        
        if answers['backmap']:
            backmap_trajectory({
                'jobname': jobname,
                'n_strides': len(flist)
            }, work_dir, answers['cg_model'],
               work_dir / f"{jobname}_nw.pdb",
               work_dir / f"{jobname}_cg_nw.dcd")

def backmap_trajectory(params: Dict[str, Any],
                      work_dir: Union[str, Path],
                      cg_model: str,
                      pdb_input: Union[str, Path],
                      traj_input: Union[str, Path]) -> Tuple[Path, Path]:
    ###Convert CG trajectory to atomistic using cg2all###
    work_dir = Path(work_dir)
    pdb_input = Path(pdb_input)
    traj_input = Path(traj_input)
    
    # Prepare output file names
    pdb_ini_output = work_dir / f"{params['jobname']}_frame0_backmapped.pdb"
    traj_output = work_dir / f"{params['jobname']}_backmapped.dcd"
    pdb_output = work_dir / f"{params['jobname']}_backmapped.pdb"
    aligned_dcd = work_dir / f"{params['jobname']}_backmapped_align.dcd"
    aligned_pdb = work_dir / f"{params['jobname']}_backmapped_align.pdb"

    # Verify cg2all is available
    try:
        subprocess.run(["convert_cg2all", "--help"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("cg2all not found. Please install it first.")

    with tqdm(total=3, desc="Backmapping") as pbar:
        # Backmap initial frame
        subprocess.run(
            f"convert_cg2all -p {pdb_input} -o {pdb_ini_output} --cg {cg_model}", 
            shell=True, check=True
        )
        pbar.update(1)
        
        # Backmap trajectory
        subprocess.run(
            f"convert_cg2all -p {pdb_input} --dcd {traj_input} "
            f"-o {traj_output} -opdb {pdb_output} "
            f"--cg {cg_model} --batch 1", 
            shell=True, check=True
        )
        pbar.update(1)
        
        # Align backmapped trajectory
        _align_trajectory(pdb_ini_output, traj_output, aligned_dcd, aligned_pdb)
        pbar.update(1)
    
    print(f"\t> Backmapped trajectory saved to {aligned_dcd}")
    return aligned_dcd, aligned_pdb

def _align_trajectory(pdb_ref: Path,
                     traj_input: Path,
                     aligned_dcd: Path,
                     aligned_pdb: Path) -> None:
    ###Helper function to align trajectory###
    u1 = mda.Universe(str(pdb_ref), str(traj_input))
    u2 = mda.Universe(str(pdb_ref), str(pdb_ref))
    u2.trajectory[0]
    
    align.AlignTraj(u1, u2, select='name CA', in_memory=True).run()
    
    with mda.Writer(str(aligned_dcd), u1.atoms.n_atoms) as W:
        for ts in u1.trajectory:
            W.write(u1.atoms)
    
    u2.atoms.write(str(aligned_pdb))

def RMSD_compute(work_dir: Union[str, Path],
                jobname: str,
                write_trajectory: int,
                stride_traj: int,
                simulation_ns: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ###Compute RMSD for CG and atomistic trajectories###
    work_dir = Path(work_dir)
    nw_dcd = work_dir / f"{jobname}_cg_nw.dcd"
    nw_pdb = work_dir / f"{jobname}_nw.pdb"
    traj_atomistic = work_dir / f"{jobname}_backmapped_align.dcd"
    pdb_ref_atomistic = work_dir / f"{jobname}_backmapped_align.pdb"

    # Load trajectories
    traj_load_cg = pt.load(str(nw_dcd), str(nw_pdb))
    traj_load_atomistic = pt.load(str(traj_atomistic), str(pdb_ref_atomistic))
    ref_top = pt.load(str(pdb_ref_atomistic), str(pdb_ref_atomistic))

    # Calculate RMSD
    rmsd_cg = pt.rmsd(traj_load_cg, ref=0)
    rmsd_atomistic = pt.rmsd(traj_load_atomistic, ref=0, mask="@CA,C,O,N,H")

    # Create time array
    time_array = _create_time_array(len(rmsd_cg), write_trajectory, stride_traj)
    
    # Plot results
    _plot_rmsd(time_array, rmsd_cg, rmsd_atomistic, simulation_ns, work_dir, jobname)
    
    return pd.DataFrame(rmsd_cg), pd.DataFrame(rmsd_atomistic)

def _create_time_array(n_frames: int,
                      write_freq: int,
                      stride: int) -> np.ndarray:
    ###Create time array for trajectory analysis###
    time = n_frames * write_freq / 1000
    return np.arange(0, time, write_freq/1000) * stride

def _plot_rmsd(time_array: np.ndarray,
              rmsd_cg: np.ndarray,
              rmsd_atomistic: np.ndarray,
              simulation_ns: float,
              output_dir: Path,
              prefix: str) -> None:
    ###Plot RMSD results###
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # CG plot
    ax1.plot(time_array, rmsd_cg, alpha=0.2, color='blue')
    ax1.plot(time_array, pd.Series(rmsd_cg).rolling(10).mean(), 
             color='blue', linewidth=1.5)
    ax1.set(xlim=(0, simulation_ns), 
            xlabel="Time (ns)", 
            ylabel="RMSD (Å)",
            title="Coarse-grained")
    
    # Atomistic plot
    ax2.plot(time_array, rmsd_atomistic, alpha=0.2, color='blue')
    ax2.plot(time_array, pd.Series(rmsd_atomistic).rolling(10).mean(),
             color='blue', linewidth=1.5)
    ax2.set(xlim=(0, simulation_ns),
            xlabel="Time (ns)",
            title="Back-mapped")
    
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_rmsd.png", dpi=600, bbox_inches='tight')
    plt.close()

def gy_radius(work_dir: Union[str, Path],
             jobname: str,
             write_trajectory: int,
             stride_traj: int,
             simulation_ns: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ###Compute radius of gyration###
    work_dir = Path(work_dir)
    nw_dcd = work_dir / f"{jobname}_cg_nw.dcd"
    nw_pdb = work_dir / f"{jobname}_nw.pdb"
    traj_atomistic = work_dir / f"{jobname}_backmapped_align.dcd"

    # Load trajectories
    traj_load_cg = pt.load(str(nw_dcd), str(nw_pdb))
    traj_load_atomistic = pt.load(str(traj_atomistic))

    # Calculate Rg
    radgyr_cg = pt.radgyr(traj_load_cg)
    radgyr_atom = pt.radgyr(traj_load_atomistic)

    # Create time array
    time_array = _create_time_array(len(radgyr_cg), write_trajectory, stride_traj)
    
    # Plot results
    _plot_gyradius(time_array, radgyr_cg, radgyr_atom, simulation_ns, work_dir, jobname)
    
    return pd.DataFrame(radgyr_cg), pd.DataFrame(radgyr_atom)

def _plot_gyradius(time_array: np.ndarray,
                 radgyr_cg: np.ndarray,
                 radgyr_atom: np.ndarray,
                 simulation_ns: float,
                 output_dir: Path,
                 prefix: str) -> None:
    ###Plot radius of gyration results###
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # CG plot
    ax1.plot(time_array, radgyr_cg, alpha=0.2, color='green')
    ax1.plot(time_array, pd.Series(radgyr_cg).rolling(10).mean(),
             color='green', linewidth=1.5)
    ax1.set(xlim=(0, simulation_ns),
            xlabel="Time (ns)",
            ylabel="Rg (Å)",
            title="Coarse-grained")
    
    # Atomistic plot
    ax2.plot(time_array, radgyr_atom, alpha=0.2, color='green')
    ax2.plot(time_array, pd.Series(radgyr_atom).rolling(10).mean(),
             color='green', linewidth=1.5)
    ax2.set(xlim=(0, simulation_ns),
            xlabel="Time (ns)",
            title="Back-mapped")
    
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_gyradius.png", dpi=600, bbox_inches='tight')
    plt.close()

def RMSF_graph(work_dir: Union[str, Path],
              jobname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ###Compute and plot RMSF###
    work_dir = Path(work_dir)
    nw_dcd = work_dir / f"{jobname}_cg_nw.dcd"
    nw_pdb = work_dir / f"{jobname}_nw.pdb"
    traj_atomistic = work_dir / f"{jobname}_backmapped_align.dcd"

    # Load trajectories
    traj_load_cg = pt.load(str(nw_dcd), str(nw_pdb))
    traj_load_atomistic = pt.load(str(traj_atomistic))

    # Calculate RMSF
    rmsf_cg = pt.rmsf(traj_load_cg, "byres")
    rmsf_atom = pt.rmsf(traj_load_atomistic, "byres")

    # Plot results
    _plot_rmsf(rmsf_cg, rmsf_atom, work_dir, jobname)
    
    return pd.DataFrame(rmsf_cg), pd.DataFrame(rmsf_atom)

def _plot_rmsf(rmsf_cg: np.ndarray,
              rmsf_atom: np.ndarray,
              output_dir: Path,
              prefix: str) -> None:
    ###Plot RMSF results###
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # CG plot
    ax1.plot(rmsf_cg[:,1], color='red')
    ax1.set(xlabel="Residue",
            ylabel="RMSF (Å)",
            title="Coarse-grained",
            xlim=(0, len(rmsf_cg[:-1])))
    
    # Atomistic plot
    ax2.plot(rmsf_atom[:,1], color='red')
    ax2.set(xlabel="Residue",
            title="Back-mapped",
            xlim=(0, len(rmsf_atom[:-1])))
    
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_rmsf.png", dpi=600, bbox_inches='tight')
    plt.close()

def pearson_CC(work_dir: Union[str, Path],
              jobname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ###Compute Pearson's cross-correlation###
    work_dir = Path(work_dir)
    nw_dcd = work_dir / f"{jobname}_cg_nw.dcd"
    nw_pdb = work_dir / f"{jobname}_nw.pdb"
    traj_atomistic = work_dir / f"{jobname}_backmapped_align.dcd"

    # Load trajectories
    traj_load_cg = pt.load(str(nw_dcd), str(nw_pdb))
    traj_load_atomistic = pt.load(str(traj_atomistic))

    # Calculate cross-correlation
    cc_cg = matrix.correl(traj_load_cg)
    cc_atom = matrix.correl(traj_load_atomistic, '@CA')

    # Plot results
    _plot_cross_correlation(cc_cg, cc_atom, work_dir, jobname)
    
    return pd.DataFrame(cc_cg), pd.DataFrame(cc_atom)

def _plot_cross_correlation(cc_cg: np.ndarray,
                          cc_atom: np.ndarray,
                          output_dir: Path,
                          prefix: str) -> None:
    ###Plot cross-correlation matrices###
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CG plot
    im1 = ax1.imshow(cc_cg, cmap='PiYG_r', vmin=-1, vmax=1, origin='lower')
    ax1.set(title="Coarse-grained",
            xlabel="Beads",
            ylabel="Beads")
    plt.colorbar(im1, ax=ax1).set_label("CCij")
    
    # Atomistic plot
    im2 = ax2.imshow(cc_atom, cmap='PiYG_r', vmin=-1, vmax=1, origin='lower')
    ax2.set(title="Back-mapped",
            xlabel="Residues",
            ylabel="Residues")
    plt.colorbar(im2, ax=ax2).set_label("CCij")
    
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_cross_correlation.png", 
               dpi=600, bbox_inches='tight')
    plt.close()
    
