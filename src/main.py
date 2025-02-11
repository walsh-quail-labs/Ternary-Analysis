import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.processing import Processing

### CONFIGURATION ###
f_config_keyPatientID = '../data/Mouse/MiceKey.xlsx'
segmentation_dir = "../../Mouse/Segmentation"
celltype_dir = "../../Mouse/MATData_converted"
save_dir = "../output/Mouse_pairwise_(Tc-Cancer-MO)"


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df_config_keyPatientID = pd.read_excel(f_config_keyPatientID)
scan_lst = df_config_keyPatientID['FileName'].values
scan_lst = scan_lst[:2]  # Limit to first 10 scans for testing

# Define requested cell types and their colors
requested_celltypes = ["Cancer", "Tc", "MO"]
node_colors = {'Cancer': 'red', 'Tc': 'blue', 'MO': 'tab:green'}
threshold_distance = 20
num_processes = 16 

### FUNCTION TO PROCESS A SINGLE SCAN ###
def process_single_scan(scan):
    """
    Processes a single scan using the Processing class.
    """
    try:
        processor = Processing(
            segmentation_dir=segmentation_dir,
            celltype_dir=celltype_dir,
            scan=scan,
            threshold_distance=threshold_distance,
            node_colors=node_colors,
            save_dir=save_dir
        )
        processor.process_scan(requested_celltypes)
    except Exception as e:
        print(f"Error processing {scan}: {e}")

### MULTIPROCESSING: PROCESS SCANS IN PARALLEL ###
if __name__ == "__main__":
    pool = mp.Pool(processes=num_processes)

    with tqdm(total=len(scan_lst)) as pbar:
        def update(*_):
            pbar.update()
        
        for _ in pool.imap_unordered(process_single_scan, scan_lst):
            update()

    pool.close()
    pool.join()
