import tkinter.filedialog as fd
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import caiman as cm
import os
import sys
from glob import glob

def main():
    gcamp_920=True #False for 2 color data
    ftype='tif'
    mag =1 #magnification
    fr=60 #frame rate to play back
    ds_ratio=.2 #downsample in time ratio (0.1 10x downsample)
    
    #images to play
    series = fd.askdirectory(initialdir=r'D:\\')

    print("Data Folder: " +series)

    if gcamp_920: #gcamp data
        print('playing tif files') #get raw images
        if gcamp_920:
            tifs = [os.path.join(series,f) for f in os.listdir(series) if (ftype in f)]
        fnames = [os.path.join(series,f) for f in tifs]
        fnames.sort()
    
        movie = cm.load_movie_chain(fnames)
        moviehandle = movie.resize(1, 1, ds_ratio).play(gain=1,q_min=1,q_max=99.5, fr=fr, magnification=mag)
    else: #mcherry data: 
        pass
        #to do - 2 color data
        

if __name__=='__main__':
    main()