import numpy as np
import argparse 
import sys 
import tifffile as tiff

def main(argv):
    parser = argparse.ArgumentParser(description='Check large tif stack dimensions')
    parser.add_argument('fname',    type=str, help='File path')
    parser.add_argument('--split_stack', action='store_true',default=False,help='split stack')
    args = parser.parse_args(argv)

    fname = args.fname

    with tiff.TiffFile(fname) as tif:
        shape = tif.series[0].shape
        print(f'{fname}  shape: {shape}')

if __name__=='__main__':
    main(sys.argv[1:]) 