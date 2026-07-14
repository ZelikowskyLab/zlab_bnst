from suite2p.run_s2p import run_s2p
from suite2p.default_ops import default_ops

# Start with defaults
ops = default_ops()

# Paths
db = {
    'data_path': [r"D:\Michael\fs_vmretro_bnst_gcamp6f_2024-05-16-13-28-33"],  # folder with your data
}

# --- Motion correction only ---
ops['do_registration'] = 1     # run motion correction
ops['nonrigid'] = True         # set False for rigid only
ops['do_segmentation'] = 0     # skip cell detection
ops['roidetect'] = False
ops['spikedetect'] = False
ops['neuropil_extract'] = False
#--- Parameters for Motion correction --- 
ops['maxregshift'] = 0.1 # Max allowed registration shift, as a fraction of frame max(width and height).
ops['maxregshiftNR'] = 10 #Maximum pixel shift allowed for nonrigid, relative to rigid, may need to increase value for unstable recordings.
ops['block_size'] = (64,64) #Block size for non-rigid registration (** keep this a multiple of 2, 3, and/or 5 **).
ops['batch_size'] = 500 # Number of frames per batch - choose fewer if running out of memory
ops['smooth_sigma'] = 5 #Gaussian smoothing in XY; ~1 good for 2P recordings, 3-5 may work well for 1P recordings
# --- Save options ---
ops['reg_tif'] = True          # save registered TIFFs
ops['nimg_init'] = 1000        # ~frames per chunked TIFF (adjust smaller if needed)

# Run
run_s2p(ops=ops, db=db)