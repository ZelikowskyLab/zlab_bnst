import pandas as pd
import h5py 
import numpy as np
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import argparse
import sys 
import os 
from utils import ridge_plot
import matplotlib
import matplotlib.pyplot as plt 
#matplotlib.use('TkAgg')
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
def main(argv):
## example in command line
    #conda activate caiman
    #python .\plot_activity.py "Y:\Michael\Miniscope_Desktop_Backup\dbnst_endo_c1_2024-08-15-11-58-11\suite2p\plane0\reg_tif\cnmfe_results.hdf5" testmouse 20240815 testsession pdf --save_traces
###    
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf_path',type=str,help='Path to hdf cnmfe results file')
    parser.add_argument('animal',type=str,help='anima name')
    parser.add_argument('session_date', type=str, help='session date in YYYYMMDD')
    parser.add_argument('session_name',type=str,help = 'session description')
    parser.add_argument('backend',type=str,help='Matplotlib plotting backend')
    parser.add_argument('--save_traces', action='store_true', default=False, help='Save traces to output')
    parser.add_argument('--savepath',type=str,help='Save path for csv file')
    args=parser.parse_args(argv)

    h5path = args.hdf_path
    animal = args.animal
    date = args.session_date
    session = args.session_name
    save = args.save_traces
    plot = args.backend
    cnmfe = load_CNMF(h5path)

    F = cnmfe.estimates.C + cnmfe.estimates.YrA
    cnmfe.estimates.detrend_df_f(quantileMin=8, 
                                       frames_window=5000, 
                                       flag_auto=True,
                                         use_fast=False, 
                                         use_residuals=True, 
                                         detrend_only=False)
    #signal = zscore(F,axis=1)
    signal = cnmfe.estimates.F_dff
    smoothed = gaussian_filter1d(signal,sigma=3)
    print('shape dff: '+str(signal.shape))

    if plot in ('agg', ' '):
        fr = cnmfe.params.data['fr']
        times = np.arange(signal.shape[1]) / fr
        fig, ax = plt.subplots(figsize=(12, 4))
        ridge_plot(smoothed, times, title=f'{animal} {date} {session}', ax=ax,
                overlay=None,
                line_width=.5,
                trace_spacing=100)
        ax.set_xlabel('Time (s)')

        if plot == 'pdf':
            pdf_path = os.path.splitext(args.savepath)[0] + '_activity.pdf'
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f'Saved plot to {pdf_path}')
        else:
            plt.show()

        if save:
            df = pd.DataFrame(signal.T, columns=[f'{i}' for i in range(signal.shape[0])])
            df.to_csv(os.path.join(args.savepath, f"{animal}_{date}_{session}_dff.csv"))
            print('saved: '+os.path.join(args.savepath, f"{animal}_{date}_{session}_dff.csv") )
        return

if __name__ == '__main__':
    #main(sys.argv[1:])
    main([r"Y:\Michael\Miniscope_Desktop_Backup\dbnst_endo_c1_2024-08-15-11-58-11\suite2p\plane0\reg_tif\cnmfe_results.hdf5", 
         'testmouse',
         '20240815', 
         'testsession', 
         'pdf', 
         '--save_traces', 
         '--savepath', r"C:\Users\ZLab\Downloads\test.csv"])

