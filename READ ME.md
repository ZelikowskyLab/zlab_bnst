# Methods for 1p dBNST preprocessing
## Software
- CaImAn
- Suite2p
- Fiji
## Scripts
```
motion_correction_s2p.py
play_movie.py 
cnmf_e.ipynb
export_to_fiji.py
```
## Pre-processing Steps
1. Perform spatial downsample (2x) and band pass filter in IDPS
2. Export tiff, edit metadata csv file
3. Open in Fiji, crop desired region, export as substacks (~2GB each)
4. Run ```motion_correction_s2p.py```. Output tifs are saved in suite2p output folder (must use suite2p conda env)
5. Run ```play_movie.py``` to evaluate motion. Can also open registered tifs in fiji and inspect. (must use caiman conda env)
6. Run ``` cnmf_e.ipynb``` for cnmfe (must use caiman conda env)
- Can do initial component inspection here
- optional ```cnmfe_report.py```
7. Run ```export_to_fiji.py``` to export rois to fiji
8. Evaluate ROIs in Fiji
9. Run ```export_traces.py``` with finalized components.

