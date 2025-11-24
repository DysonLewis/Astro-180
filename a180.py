# code for Astronomy 180 lab exercises
#
# Michael Fitzgerald (mpfitz@ucla.edu) 2013-10-6

import numpy as np

def fit_abs_line(x, y, sp=None):
    """
    Fit an absorption line to a 1-d spectrum.

    Syntax:
      fit_y, parms = fit_abs_line(x, y, sp=None)

    fit_y    The model profile.
    parms    An array of parameters:  [x0, sigma, A, B, C, D]
    sp       An optional array of starting parameters

    The fit is performed such that
       fit_y = B + Cx + Dx^2 - A/sqrt(2 pi)/sigma * exp(-(x-x0)^2 / 2 / sigma^2)
    """
    
    # verify input
    if len(x) != len(y):
        print("ERROR:  Require inputs have equal length")
        raise ValueError
    if len(y.shape) != 1:
        print("ERROR:  require 1-dimensional inputs")
        raise ValueError

    # choose starting parameters
    if sp is None:
        B = y.max()
        ym = y.min()
        x0 = x[y.argmin()]
        hp = (B+ym)/2. # half-power points
        sd = np.sign(y-hp)
        xhp = x[np.nonzero(sd[1:] != sd[:-1])] # locations of sign change
        assert len(xhp) == 2 # should only cross half-power points twice
        fwhm = xhp[1]-xhp[0]
        sig = fwhm/2.3548
        A = (B-ym)*np.sqrt(2.*np.pi)*sig
        sp = np.array([x0, sig, A, B, 0., 0.])

    def model_fn(p):
        x0, sig, A, B, C, D = p
        return B + C*x + D*x**2 - \
               A/np.sqrt(2.*np.pi)/sig * np.exp(-(x-x0)**2/2./sig**2)
    fit_fn = lambda p: y-model_fn(p)

    # perform fit
    from scipy.optimize import leastsq
    p_opt, ier = leastsq(fit_fn, sp.copy())
    # check output
    assert ier in (1, 2, 3, 4)

    return model_fn(p_opt), p_opt



# original code in pixwt.c by Marc Buie
# 
# ported to pixwt.pro (IDL) by Doug Loucks, Lowell Observatory, 1992 Sep
#
# subsequently ported to python by Michael Fitzgerald,
# LLNL, fitzgerald15@llnl.gov, 2007-10-16
#

def _arc(x, y0, y1, r):
    """
    Compute the area within an arc of a circle.  The arc is defined by
    the two points (x,y0) and (x,y1) in the following manner: The
    circle is of radius r and is positioned at the origin.  The origin
    and each individual point define a line which intersects the
    circle at some point.  The angle between these two points on the
    circle measured from y0 to y1 defines the sides of a wedge of the
    circle.  The area returned is the area of this wedge.  If the area
    is traversed clockwise then the area is negative, otherwise it is
    positive.
    """
    return 0.5 * r**2 * (np.arctan(y1/x) - np.arctan(y0/x))

def _chord(x, y0, y1):
    """
    Compute the area of a triangle defined by the origin and two
    points, (x,y0) and (x,y1).  This is a signed area.  If y1 > y0
    then the area will be positive, otherwise it will be negative.
    """
    return 0.5 * x * (y1 - y0)

def _oneside(x, y0, y1, r):
    """
    Compute the area of intersection between a triangle and a circle.
    The circle is centered at the origin and has a radius of r.  The
    triangle has verticies at the origin and at (x,y0) and (x,y1).
    This is a signed area.  The path is traversed from y0 to y1.  If
    this path takes you clockwise the area will be negative.
    """

    if np.all((x==0)): return x

    sx = x.shape
    ans = np.zeros(sx, dtype=float)
    yh = np.zeros(sx, dtype=float)
    to = (abs(x) >= r)
    ti = (abs(x) < r)
    if np.any(to):
        ans[to] = _arc(x[to], y0[to], y1[to], r)
    if not np.any(ti):
        return ans

    yh[ti] = np.sqrt(r**2 - x[ti]**2)

    i = ((y0 <= -yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], -yh[j], r) + \
                     _chord(x[j], -yh[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], -yh[j], r) + \
                     _chord(x[j], -yh[j], yh[j]) + \
                     _arc(x[j], yh[j], y1[j], r)

    i = ((y0 > -yh) & (y0 < yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], -yh[j]) + \
                     _arc(x[j], -yh[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], yh[j]) + \
                     _arc(x[j], yh[j], y1[j], r)
        
    i = ((y0 >= yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], yh[j], r) + \
                     _chord(x[j], yh[j], -yh[j]) + \
                     _arc(x[j], -yh[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], yh[j], r) + \
                     _chord(x[j], yh[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], y1[j], r)
        
    return ans

def _intarea(xc, yc, r, x0, x1, y0, y1):
    """
    Compute the area of overlap of a circle and a rectangle.
      xc, yc  :  Center of the circle.
      r       :  Radius of the circle.
      x0, y0  :  Corner of the rectangle.
      x1, y1  :  Opposite corner of the rectangle.
    """
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return _oneside(x1, y0, y1, r) + _oneside(y1, -x1, -x0, r) + \
           _oneside(-x0, -y1, -y0, r) + _oneside(-y0, x0, x1, r)

def pixwt(xc, yc, r, x, y):
    """
    Compute the fraction of a unit pixel that is interior to a circle.
    The circle has a radius r and is centered at (xc, yc).  The center
    of the unit pixel (length of sides = 1) is at (x, y).

    Divides the circle and rectangle into a series of sectors and
    triangles.  Determines which of nine possible cases for the
    overlap applies and sums the areas of the corresponding sectors
    and triangles.

    area = pixwt( xc, yc, r, x, y )

    xc, yc : Center of the circle, numeric scalars
    r      : Radius of the circle, numeric scalars
    x, y   : Center of the unit pixel, numeric scalar or vector
    """
    return _intarea(xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5)


# -------

def ap_phot(im, x, y, rad, sky_in=None, sky_out=None, gain=1.):
    """
    Computes circular aperture photometry, with optional sky annulus.

    phot, phot_err = ap_phot(im, x, y, ad, sky_in=None, sky_out=None, gain=1.)

    Inputs:
      im       [DN]     image data
      x        [pix]    x coordinate of aperture center (im[y,x])
      y        [pix]    y coordinate of aperture center (im[y,x])
      rad      [pix]    aperture radius
      sky_in   [pix]    inner radius of sky annulus
      sky_out  [pix]    outer radius of sky annulus
      gain     [e-/DN]  gain

    Outputs:
      phot     [DN]     aperture photometry
      phot_err [DN]     1-sigma error in aperture photometry

    """

    ny, nx = im.shape

    if sky_in is not None:
        assert sky_out > sky_in
        assert sky_in >= rad
        out_rad = sky_out+2.
    else:
        out_rad = rad+2.

    # validate input
    assert rad > 0.
    assert x-out_rad >= 0.
    assert x+out_rad < nx
    assert y-out_rad >= 0.
    assert y+out_rad < ny


    # get thumbnail image
    nn = int(2*out_rad)
    by, bx = int(np.floor(y-out_rad)), int(np.floor(x-out_rad))
    ty, tx = np.mgrid[0:nn,0:nn]
    py, px = y-by, x-bx
    tim = im[by:by+nn,bx:bx+nn]


    # get aperture photometry
    p = pixwt(px, py, rad, tx, ty)
    phot = (p*tim).sum() # [DN]

    phot_var = phot/gain # [DN^2]

    if sky_in is not None:
        # get average sky brightness
        ps = pixwt(px, py, sky_out, tx, ty) - \
             pixwt(px, py, sky_in, tx, ty)
        bg_phot = (ps*tim).sum() / ps.sum() # [DN/pix]

        wbg = np.nonzero(ps)
        bg_var = tim[wbg].std()**2 # [DN^2]
        mbg_var = bg_var / ps.sum() # [DN^2]

        # subtract from flux measurement
        phot -= bg_phot*p.sum() # [DN]
        phot_var += p.sum()*bg_var + mbg_var*p.sum()**2

    phot_err = np.sqrt(phot_var)

    return phot, phot_err
