from scipy.spatial import cKDTree
import numpy as np

def match_stars(x1,y1,mag1,mag1_err,x2,y2,mag2,mag2_err,sep=2.0,debug=False):
    '''
    Match two star lists that are already close together in position.
    Returns only stars in list 1 that also matches stars in list 2 within
    some radius
    
    Inputs
    ------
    x1,y1,mag1,mag1_err  - x position, y position, magnitude, and mag error for list 1
    x2,y2,mag2,mag2_err  - x position, y position, magnitude, and mag error for list 2
    
    Keywords
    --------
    sep - only accept as a match stars that are within this distance (default: 2)
    debug - set to print out extra information (default: False)
    
    Returns
    -------
    a [8,n] numpy array where n is the number of overlapping stars within distance sep
    [0,:] - x1_matched
    [1,:] - y1_matched
    [2,:] - mag1_matched
    [3,:] - mag1_err_matched
    [4,:] - x2_matched
    [5,:] - y2_matched
    [6,:] - mag2_matched
    [7,:] - mag2_err_matched    
    '''
    
    # build a KDTree from one of the star lists of positions
    tree = cKDTree(np.transpose([x1,y1]))
    
    # use the KDTree to find the cloest star. This part will return the 
    # distance and index of the closest star in list 1 for every star in list 2
    dd,ii = tree.query(np.transpose([x2,y2]),k=1)
    
    # create new array that have matching stars in list 1 to list 2
    x1_matched = np.array(x1)[ii]
    y1_matched = np.array(y1)[ii]
    mag1_matched = np.array(mag1)[ii]
    mag1_err_matched = np.array(mag1_err)[ii]
    
    if debug:
        print(x1_matched[0:10],y1_matched[0:10])
        print(x2[0:10],y2[0:10])
        print(dd[0:10])
    
    good = np.where(dd < sep)[0]
    
    x1_matched = x1_matched[good]
    y1_matched = y1_matched[good]
    mag1_matched = mag1_matched[good]
    mag1_err_matched = mag1_err_matched[good]
    
    x2_matched = np.array(x2)[good]
    y2_matched = np.array(y2)[good]
    mag2_matched = np.array(mag2)[good]
    mag2_err_matched = np.array(mag2_err)[good]
    
    return np.array([x1_matched,y1_matched,mag1_matched,mag1_err_matched,
            x2_matched,y2_matched,mag2_matched,mag2_err_matched])
