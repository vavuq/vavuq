#!/usr/bin/env python

"""
VAVUQ (Verification And Validation and Uncertainty Quantification) can
be used as a general purpose program for verification, validation, and
uncertainty quantification. The motivation for the creation of and 
continued development of the program is to provide a cost effective and
easy way to assess the quality of computational approximations. The hope
is that the methods used in this program can be applied to fields such 
as civil engineering where they are currently often underutilized. This
code was created through efforts from Bombardelli's group and Dr. Bill 
Fleenor at the University of California Davis. The creators belong to 
the Civil & Environmental Engineering (CEE) Department, the Center for
Watershed Sciences (CWS), and the Delta Solution Team (DST). The main 
code development is headed by Kaveh Zamani and James E. Courtney.

==============================================================================
Reference: please see "Verification and Validation in Scintific
Computing" by William L. Oberkampf and Chistopher J. Roy. Chapter 8, or
Zamani, K., Bombardelli, F. A. (2014) "Analytical solutions of nonlinear
and variable-parameter transport equations for verification of numerical
solvers." Environmental Fluid Mechanics.
==============================================================================
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# This code uses: Scipy, Pandas, Numpy, Math, Matplotlib, Tkinter

# plot options [1:on, 0:off]
popt = [None]*8
popt[0] = 0 # 3D surface plot (user interaction)
popt[1] = 0 # Triangular 2D surface plot
popt[2] = 0 # Interpolated triangular 2D surface plot
popt[3] = 0 # Image plot (suppressed if unstructured mesh)
popt[4] = 0 # 3D scatter plot (user interaction)
popt[5] = 1 # Rect surface plot
popt[6] = 0 # 2D scatter
popt[7] = 0 # 2D Contour

# execute plot options
def plotmain(popt,x,y,z,titl):
    if popt[0]: surfplot(x,y,z,titl)
    if popt[1]: trisurfplot(x,y,z,titl)
    if popt[2]: triconplot(x,y,z,titl)
    if popt[3]: plotim(x,y,z,titl)
    if popt[4]: scatplot(x,y,z,titl)
    if popt[5]: sqsurfplot(x,y,z,titl)
    if popt[6]: scat2d(x,y,z,titl)
    if popt[7]: cont2d(x,y,z,titl)

# This function is based on Equation 8.73 in 'Verification and Validation
# in Scientific Computing' by W. Oberkampf and C. Roy on page 320.
# NCS: No Converged Solution
def calc_con(Lcoarse, Lfine, rc, rf):
    import numpy as np
    import math
    p = 0.5
    AA = math.log10(rf*rc)
    if Lfine == 0.:
        return 0. # NCS
    BB = abs(Lcoarse/Lfine)
    if math.isnan(BB):
        return 0. # NCS
    i = 1
    while True:
        if i > 10000:
            return 0. # NCS
        i += 1
        x = (rf**p - 1.0)*BB + rf**p
        if x < 0.:
            return 0. # NCS
        o = p
        p = math.log10(x)/AA
        if abs(o-p) < 1e-12:
            break
    return p


'''
The coarse grid convergence index (GCI) is calculated based on
Roache, P.J. 1994. "Perspective: A Method for Uniform Reporting
of Grid Refinement Studies. Journal of Fluids Engineering.
September 1994, Vol. 116/405"
  fe: fractional error
  f1: fine grid solution
  f2: coarse gride solution
   r: refinment
   p: order of accuracy'''
def GCI_coarse(f1,f2,r,p):
    return r**p * GCI_fine(f1,f2,r,p)

'''
The fine grid convergence index (GCI) is calculated based on
ASME V&V 20-2009
'''
def GCI_fine(f1,f2,r,p,fs):
    # calculate extrapolated value
    ex = (r**p*f1 - f2) / (r**p - 1.)
    # calculate error estimate
    ee = abs((f1 - f2) / (f1 + 1e-12))
    # calculate extrapolated relative error
    er = abs((ex - f1) / (ex + 1e-12))
    # calculate GCI
    GCI = (fs*ee) / (r**p - 1.)
    return (GCI,er)

"""
Calculates Statistics Validation
----------------------------------------------------
M is model data or prediction data
O is benchmark or measured data (observed data)
Paper: "Comparison of different effciency criteria for hydrological
model assesment by Krause, Boyle and Base (2005)
Nash_Sutcliffe efficiency E Nash, J. E. and J. V. Sutcliffe (1970),
River flow forecasting through conceptual models part I -A discussion
of principles, Journal of Hydrology, 10 (3), 282-290
"""
def statistics_validation(M,O):
    import numpy as np
    from scipy.stats import ks_2samp, chisquare
    M, O = M.ravel(), O.ravel()
    wn = ''
    if abs(np.mean(M)) < 0.001:
        M, O = M + 1., O + 1.
        eo = '***Results are shifted one unit to avoid division by zero***'
        wn = ''.join([wn, eo+'\n'])
    bias = np.sum(M - O) / O.size
    rms = np.sqrt(np.mean(np.square(M - O)))
    SI = rms/np.mean(M)
    x2 = O - np.mean(M)
    y2 = O - np.mean(O)
    R2 = (np.sum(np.multiply(x2, y2)) / \
         (np.sum(np.square(x2)) * np.sum(np.square(y2)))**0.5)**2
    E = 1. - np.sum(np.square(M-O)) / np.sum(np.square(y2))
    KS = ks_2samp(M.A.squeeze(),O.A.squeeze())
    CH = chisquare(M.A.squeeze(), f_exp = O.A.squeeze())
    eo = 'Bias = %(bias)s \n' \
         'Scatter Index = %(SI)s \n' \
         'RMSE = %(rms)s \n' \
         'Coefficient of Determination = %(R2)s \n' \
         'NSE = %(E)s \n' % locals()
    eo += 'K-S(stat, p-value) = (%0.4e, %0.4e)\n' % (KS[0],KS[1])
    eo += 'Chi Sq(stat, p-value) = (%0.4e, %0.4e)' % (CH[0],CH[1])
    wn = ''.join([wn, eo+'\n'])
    return wn

# read input file
def odat(fname):
    import numpy as np
    ex = fname.split('.')[-1].lower()
    try:
        if (ex == 'xlsx') | (ex == 'xls'):
            from pandas import ExcelFile
            with ExcelFile(fname) as sn:
                x = [np.matrix(sn.parse(i, header=None).values)
                     for i, n in enumerate(sn.sheet_names)]
        elif ex == 'h5':
            from pandas import HDFStore, read_hdf
            with HDFStore(fname) as df:
                x = [np.matrix(read_hdf(fname, k).values) for k in df.keys()]
        elif ex == 'csv':
            from pandas import read_csv
            df = read_csv(fname, header=None)
            x = []
            j = 0
            for i in range(len(df.columns)/3):
                z = np.asarray(df.values)[:,j:j+3]
                z = z[~np.any(np.isnan(z), axis=1)]
                x.append(np.matrix(z))
                j += 3
        elif (ex == 'txt') | (ex == 'dat'):
            with open(fname) as fn:
                icom = fn.readline()
                if ',' in icom:
                    com = True
                else:
                    com = False
            with open(fname) as fn:
                if not com:
                    from pandas import read_table
                    df = read_table(fn, sep='\t', header=None)
                else:
                    from pandas import read_csv
                    df = read_csv(fn, header=None)
                x = []
                j = 0
                for i in range(len(df.columns)/3):
                    z = np.asarray(df.values)[:,j:j+3]
                    z = z[~np.any(np.isnan(z), axis=1)]
                    x.append(np.matrix(z))
                    j += 3
    except:
        wrng('\nFile format error during read',rw)
    return x

# structured interpolation
def inter_fun(x,y,z,XI,YI,interp_method):
    import numpy as np
    XI, YI = np.asarray(XI), np.asarray(YI)
    x, y = (w.ravel().tolist()[0] for w in [x, y])
    if interp_method == 'Spline':
        from scipy.interpolate import RectBivariateSpline
        x, y = np.sort(np.unique(x)), np.sort(np.unique(y))
        if len(x) != len(y): spl = RectBivariateSpline(x,y,z.H)
        else: spl = RectBivariateSpline(x,y,z)
        imat = spl.ev(XI, YI)
    elif interp_method == 'Cubic':
        from scipy.interpolate import griddata
        z = z.ravel().tolist()[0]
        imat = griddata((x, y), z, (XI, YI), method = 'cubic')
    elif interp_method == 'Linear':
        from scipy.interpolate import griddata
        z = z.ravel().tolist()[0]
        imat = griddata((x, y), z, (XI, YI), method = 'linear')
    return imat

# unstructured interpolation
def inter_fun_un(x,y,z,XI,YI,interp_method):
    import numpy as np
    XI, YI = np.asarray(XI), np.asarray(YI)
    x, y = (w.ravel().tolist()[0] for w in [x, y])
    if interp_method == 'Spline':
        from scipy.interpolate import SmoothBivariateSpline
        spl = SmoothBivariateSpline(x,y,z)
        imat = spl.ev(XI, YI)
    elif interp_method == 'Cubic':
        from scipy.interpolate import Rbf
        rbf = Rbf(x, y, z, function='cubic')
        imat = rbf(XI, YI)
    elif interp_method == 'Linear':
        from scipy.interpolate import Rbf
        rbf = Rbf(x, y, z, function='linear')
        imat = rbf(XI, YI)
    return imat

# return number of unique values
def unval(x):
    import numpy as np
    return len(np.unique(x.ravel().tolist()[0]))

# image plot
def plotim(x,y,z,titl):
    import matplotlib.pyplot as plt
    im = plt.imshow(z, cmap='jet', interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.title(titl)
    titl += '(img).png'
    plt.savefig(titl, bbox_inches='tight')
    plt.close()

# surface plot
def sqsurfplot(x,y,z,titl):
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import warnings
    fig = plt.figure(facecolor="white")
    warnings.simplefilter("ignore")
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.5, antialiased=True)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #if np.max(z) < 0.0001:
    #    ax.zaxis.set_major_formatter(FormatStrFormatter('%.03e'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    #ax.set_zlabel('Head (m)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    with warnings.catch_warnings():
        plt.title(titl)
        plt.tight_layout()
        plt.show()
        plt.close()

# triange surface plot
def surfplot(x,y,z,titl):
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1))))
             for w in [x,y,z])
    fig = plt.figure()
    warnings.simplefilter("ignore")
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    with warnings.catch_warnings():
        plt.title(titl)
        plt.tight_layout()
        plt.show()
        plt.close()

# scatter plot
def scatplot(x,y,z,titl):    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import warnings
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1)))) 
             for w in [x,y,z])
    fig = plt.figure(facecolor="white")
    warnings.simplefilter("ignore")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap=mpl.cm.gray)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    with warnings.catch_warnings():
        plt.title(titl)
        plt.tight_layout()
        plt.show()
        plt.close()

# trisurface plot
def trisurfplot(x,y,z,titl):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np
    from matplotlib import cm
    from matplotlib.tri import Triangulation
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1)))) 
             for w in [x,y,z])
    triang = tri.Triangulation(x, y)     
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.tripcolor(triang, z, shading='flat', cmap=plt.cm.rainbow,
                  edgecolors='k')
    plt.colorbar()
    plt.title(titl)
    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig(titl+'(tri).png', bbox_inches='tight')
    plt.close()

# interpolated trisurface plot
def triconplot(x,y,z,titl):    
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings
    from matplotlib import cm
    from matplotlib.tri import Triangulation
    from matplotlib.tri import UniformTriRefiner
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1)))) 
             for w in [x,y,z])
    triang = Triangulation(x, y)
    refiner = UniformTriRefiner(triang)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(triang, lw=0.5, color='black')
    levels = np.arange(min(z), max(z), (max(z)-min(z))/100.)
    cmap = cm.get_cmap(name='jet', lut=None)
    plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
    plt.colorbar()
    plt.title(titl)
    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig(titl+'(interp).png', bbox_inches='tight')
    plt.close()
    
# 2D scatter plot
def scat2d(x,y,z,titl):
    import matplotlib.pyplot as plt
    import numpy as np
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1))))
             for w in [x,y,z])
    plt.scatter(x, y)
    plt.show()  

# 2D contour plot (structured)
def cont2d(x,y,z,titl):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import warnings
    from scipy.interpolate import griddata
    XI, YI = np.linspace(x.min(),x.max(),100),np.linspace(y.min(),y.max(),100)
    XI, YI = np.meshgrid(XI, YI)  
    x,y,z = (np.squeeze(np.asarray(np.reshape(w,(w.size,1))))
             for w in [x,y,z])
    warnings.simplefilter("ignore")
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    with warnings.catch_warnings():
        imat = griddata( (x, y), z, (XI, YI), method = 'cubic')
        CS = plt.contour(XI, YI, imat)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(titl)
        plt.show()
    
# calculate convergence accross entire mesh
def matcon(errc,errf,rc,rf):
    import numpy as np
    con = np.matrix([[calc_con(errc[i,j],errf[i,j],rc,rf)
                for j in range(errc.shape[1]) ] for i in range(errc.shape[0])])
    return con

#check number of elements in mesh
class ArrayLengthError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
def arrdimch(ny,nx,x,det):
    mul = ny*nx
    al = len(x)
    if mul != al:
        eo = '%(det)s Mesh: Number of unique x and y values ' \
             '(%(nx)s, %(ny)s) do not multiply to the array length of the ' \
             'input. \n (%(mul)s /= %(al)s)!' % locals()
        raise ArrayLengthError(eo)

# cut values outside of analysis area
def excut(x,y,p,xmin,xmax,ymin,ymax):
    import numpy as np
    exc = np.where((x<xmin) | (x>xmax) | (y<ymin) | (y>ymax))
    exc = np.unique(np.asarray(exc))    
    x,y,p = (np.delete(w,exc,0) for w in [x,y,p])
    return x,y,p


################
# Main Program #
################
def Validation(pd2, pd1): # Validation Calculations
    global var4_1
    import numpy as np
    from numpy import linalg as LA
    
    i = 0
    x = odat(fname)
    num = x[0] # 1st tab: numerical data
    benchmark = x[1] # 2nd tab: benchmark data
    
    eo = 'Validation data is loaded: %s' % (fname,)
    i += 1
    # text format tags
    txt.tag_config("n", background="#e7efff", foreground="red")
    txt.tag_config("a", foreground="black")
    txt.tag_config("w", background="#ffffff", foreground="red")
    label = tk.Label(root, text="T")
    font = tkFont.Font(font=label['font'])
    txt.tag_config("b", foreground="black",
                   font=(font.actual()['family'],
                         '-'+str(font.actual()['size']),"bold"))
    txt.tag_raise("sel")
    txt.insert(str(i)+'.0',eo+'\n', ("n", "a"))
    txt.mark_set(tk.INSERT,str(i)+'.0')
    
    if np.asarray(num).shape[1] > 1:
        num,benchmark = (np.asarray(w).ravel() for w in [num,benchmark])
    
    # full
    i += 1
    txt.insert(str(i)+'.0','***Full***\n', ("w", "b"))
    eo = statistics_validation(num,benchmark)
    i += 1
    txt.insert(str(i)+'.0',eo)
    
    if var4_1.get() == 'Yes':
        def ctwt(url,num,benchmark,ndat,fmt): # calc and write sections
            url = np.unique(np.asarray(url))
            if (url.shape[0] >= 0) & (url.shape[0] < len(num)):
                eo = '%s of %s data points used' % (len(num)-url.shape[0],
                                                    len(num))
                if fmt:
                    txt.insert(tk.END,eo,fmt)
                else:
                    txt.insert(tk.END,eo)
                urb,urn = (np.delete(w,url) for w in [benchmark,num])
                eo = statistics_validation(urn,urb)
                if fmt:
                    txt.insert(tk.END,'\n'+eo,fmt)
                else:
                    txt.insert(tk.END,'\n'+eo)
            else:
                if fmt:
                    txt.insert(tk.END,ndat+'\n',fmt)
                else:
                    txt.insert(tk.END,ndat+'\n')                    
        brk = 16*'-'+'\n'
        # middle
        txt.insert(tk.END,'\n')
        txt.insert(tk.END,'***Middle***\n', ("n", "b"))
        rng = max(max(benchmark),max(num)) - min(min(benchmark),min(num))
        urc = min(min(benchmark),min(num)) + pd1*rng
        lrc = min(min(benchmark),min(num)) + pd2*rng
        url = np.where((benchmark < urc) | (benchmark > lrc))
        ndat = 'There are no data in this range to analyze'  
        ctwt(url,num,benchmark,ndat,("n", "a"))
        # upper
        txt.insert(tk.END,'\n',("n", "a"))
        txt.insert(tk.END,'***Upper***\n', ("w", "b"))
        lrc = min(min(benchmark),min(num)) + pd2*rng
        url = np.where(benchmark <= lrc)
        ctwt(url,num,benchmark,ndat,False)
        # lower
        txt.insert(tk.END,'\n')
        txt.insert(tk.END,'***Lower***\n', ("n", "b"))
        urc = min(min(benchmark),min(num)) + pd1*rng
        url = np.where(benchmark >= urc)
        ctwt(url,num,benchmark,ndat,("n", "a"))
        txt.insert(tk.END,'\n',("n", "a"))


def ValidationPlt(): # Validation Plots
    global Obwc,Obmc,Obvc    
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = odat(fname)
    num = x[0]# 1st tab: numerical data
    benchmark = x[1] # 2nd tab: benchmark data
    
    if np.asarray(num).shape[1] > 1:
        num,benchmark= (np.asarray(w).ravel() for w in [num,benchmark]) 
    
    with plt.style.context(('grayscale')):
        # plot Numerical and Benchmark data
        if Obwc.get():
            fig, ax = plt.subplots()
            ax.plot(num, label='Numerical')
            ax.plot(benchmark, label='Benchmark')
            plt.xlabel('Data Points')
            plt.ylabel('State')
            ax.set_xlim((0, len(num)))
            ax.set_ylim((min(min(num),min(benchmark)),max(max(num),
                        max(benchmark))))
            plt.legend(prop={'size':12})
            #plt.legend()
            plt.show()
        # plot Benchmark verus Numerical
        if Obvc.get():
            fig, ax = plt.subplots()
            ax.plot([0,max(max(num),max(benchmark))],[0,max(max(num),
                    max(benchmark))],'r')
            ax.scatter(benchmark,num, s=10)
            ax.set_xlim((min(benchmark), max(benchmark)))
            ax.set_ylim((min(num),max(num)))
            plt.xlabel('Benchmark')
            plt.ylabel('Numerical')
            plt.show()
        # plot Benchmark minus Numerical data
        if Obmc.get():
            fig, ax = plt.subplots()
            ax.plot(benchmark-num, '#0072B2', label='Benchmark - Numerical')
            plt.xlabel('Data Points')
            plt.ylabel('State Difference')
            ax.set_xlim((0, len(num)))
            ax.set_ylim((min(benchmark-num),max(benchmark-num)))
            plt.legend(prop={'size':12})
            #plt.legend()
            plt.show()


def Verification(interp_method,log_flag,runopt,*args): # Verification
    global imat,book,Surf,Scat,Cont,Imag,popt,varc1,varc2,varc3,varc4,specifar
    import numpy as np
    import math
    import xlwt
    from numpy import linalg as LA

    if 'figures' in log_flag:
        popt = [0]*len(popt)
        if Cont.get(): popt[7] = 1
        if Surf.get(): popt[5] = 1
        if Scat.get(): popt[4] = 1
        if Imag.get(): popt[3] = 1
    
    # text format tags
    txt.tag_config("n", background="#e7efff", foreground="red")
    txt.tag_config("a", foreground="black")
    txt.tag_config("w", background="#ffffff", foreground="red")
    label = tk.Label(root, text="T")
    font = tkFont.Font(font=label['font'])
    txt.tag_config("b", foreground="black",
                   font=(font.actual()['family'],
                         '-'+str(font.actual()['size']),"bold"))
    txt.tag_raise("sel")
  
    eo = 'Interpolation method is: %(interp_method)s \n' \
         'The method for the results is: %(log_flag)s' % locals()
    txt.insert('1.0',eo+'\n',("n", "a"))
    txt.mark_set(tk.INSERT,'1.0')
    
    if 'Uncertainty' in runopt:
        if   ('95' in args[0]) & ('Str' in args[0]): fs = 1.25
        elif ('95' in args[0]) & ('Uns' in args[0]): fs = 3.0
        elif ('99' in args[0]) & ('Str' in args[0]): fs = 1.65
        elif ('99' in args[0]) & ('Uns' in args[0]): fs = 4.0
        eo = 'Factor of safety: %s \n' % (fs,)
        txt.insert(tk.END,eo,("n", "a"))

    # load data
    x = odat(fname)
    eo = 'Data is loaded: %s \n' % (fname,)
    txt.insert(tk.END,eo)

    # sort mesh qualities    
    smq = [(i, x[i].shape[0]) for i in range(3)]
    smq = sorted(smq, key=lambda shape: shape[1])
    coarse,mid,fine = [x[smq[i][0]] for i in range(3)]

    # manage data
    xc,zc,pc = (coarse[:,w] for w in range(3))
    xm,zm,pm = (mid[:,w] for w in range(3))
    xf,zf,pf = (fine[:,w] for w in range(3))

    # record exact solution
    if 'Code' in runopt:
        pce, pme, pfe = (w[:,3] for w in [coarse,mid,fine])
        pc,pm,pf = pc-pce, pm-pme, pf-pfe

    hc,hf = (np.sqrt(1. / len(w)) for w in [xc,xf])

    # convert integer values to floats
    def conf(*args):
        for w in args:
            yield w.astype('float')
    xc,zc,pc = conf(xc,zc,pc)
    xm,zm,pm = conf(xm,zm,pm)
    xf,zf,pf = conf(xf,zf,pf)

    # exclue values outside of user specified ranges
    if specifar.get():
        xc,zc,pc = excut(xc,zc,pc,varc1.get(),varc2.get(),varc3.get(),
                         varc4.get())
        def vspa(x):
            x = np.unique(x.ravel().tolist()[0])
            x = abs(x[1] - x[0])
            return x
        xsp,zsp = vspa(xm),vspa(zm)
        xm,zm,pm = excut(xm,zm,pm,varc1.get()-xsp,varc2.get()+xsp,
                         varc3.get()-zsp,varc4.get()+zsp)
        xsp,zsp = vspa(xf),vspa(zf)
        xf,zf,pf = excut(xf,zf,pf,varc1.get()-xsp,varc2.get()+xsp,
                         varc3.get()-zsp,varc4.get()+zsp)

    # determin number of unique values    
    num_c_x,num_c_y,num_m_x,num_m_y,num_f_x,num_f_y \
            = (unval(w) for w in [xc,zc,xm,zm,xf,zf])
      
    # check if unstructured
    ustruc = False
    if (num_c_x * num_c_y > len(xc) and num_m_x * num_m_y > len(xm) and
            num_f_x * num_f_y > len(xf)):
        ustruc = True
        eo = 'Unstructured Input Mesh'
        txt.insert(tk.END,'\n'+eo,("n", "a"))
        # don't plot image
        popt[3] = 0
    else:
        eo = 'Structured Input Mesh'
        txt.insert(tk.END,eo+'\n',("n", "a"))

    if ustruc:
        rf = (float(num_m_x)/float(num_c_x))**(1./2.)
        rc = (float(num_f_x)/float(num_m_x))**(1./2.)
    else: # refinement ratios by mesh spacing
        # calculate rc        
        x,y = (np.unique(w.ravel().tolist()[0]) for w in [xc,xm])
        rc = abs(x[1]-x[0])/abs(y[1]-y[0])
        x,y = (np.unique(w.ravel().tolist()[0]) for w in [zc,zm])
        rc = (rc*abs(x[1]-x[0])/abs(y[1]-y[0]))**0.5
        # calculate rf        
        x,y = (np.unique(w.ravel().tolist()[0]) for w in [xm,xf])
        rf = abs(x[1]-x[0])/abs(y[1]-y[0])        
        x,y = (np.unique(w.ravel().tolist()[0]) for w in [zm,zf])
        rf = (rf*abs(x[1]-x[0])/abs(y[1]-y[0]))**0.5
        # refinement ratios by numbers of nodes in meshes
        #rf = math.sqrt(float(num_f_x)*float(num_f_y)/float(num_m_x)/ \
                                                    #float(num_m_y))
        #rc = math.sqrt(float(num_m_x)*float(num_m_y)/float(num_c_x)/ \
                                                    #float(num_c_y))

    eo = 'rc = %(rc)s\nrf = %(rf)s' % locals()
    txt.insert(tk.END,eo+'\n')

    # this limit (4/3) is suggested by ASME V&V 20-2009, Roache and Ghia
    if rf<4./3. or rc<4./3.:
        eo = 'Refinment ratio must be greater than 4/3! \n' \
             'rc =%(rc)0.3f\n' \
             'rf =%(rf)0.3f' % locals()
        txt.insert(tk.END,eo+'\n')

    if not ustruc: # arrange into matracies
        # coarse mesh
        arrdimch(num_c_y, num_c_x, xc,'Coarse')
        xc,zc,pc = (np.reshape(w,(num_c_y,num_c_x)) for w in [xc,zc,pc])
        # medium mesh
        arrdimch(num_m_y, num_m_x, xm,'Medium')
        xm,zm,pm = (np.reshape(w,(num_m_y,num_m_x)) for w in [xm,zm,pm])
        # fine mesh
        arrdimch(num_f_y, num_f_x, xf,'Fine')
        xf,zf,pf = (np.reshape(w,(num_f_y,num_f_x)) for w in [xf,zf,pf])

    if 'figures' in log_flag:
        plotmain(popt,xc,zc,pc,'Coarse mesh')
        plotmain(popt,xm,zm,pm,'Medium mesh')
        plotmain(popt,xf,zf,pf,'Fine mesh')

    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Coarse mesh")
    for i in range(xc.shape[0]):
        for j in range(xc.shape[1]):
            sheet1.write(i, j, pc[i,j])

    XI, YI = np.asarray(xc), np.asarray(zc)

    #interpolate refined calculations to locations of coarse mesh
    if ustruc:
        pmc = inter_fun_un(xm,zm,pm,XI,YI,interp_method)
        if runopt == 'Code':
            pfc = inter_fun_un(xf,zf,pf,xm,zm,interp_method)
        else:
            pfc = inter_fun_un(xf,zf,pf,XI,YI,interp_method)
    else:
        pmc = inter_fun(xm,zm,pm,XI,YI,interp_method)
        if runopt == 'Code':
            pfc = inter_fun(xf,zf,pf,xm,zm,interp_method)
        else:
            pfc = inter_fun(xf,zf,pf,XI,YI,interp_method)

    if 'figures' in log_flag:
        ptx = 'Interpolations Corresponding to'
        plotmain(popt,xc,zc,pmc,'Mid %s Coarse' % (ptx,))
        if runopt == 'Code':
            plotmain(popt,xm,zm,pfc,'Fine %s Mid' % (ptx,))
        else:
            plotmain(popt,xc,zc,pfc,'Fine %s Coarse' % (ptx,))

    sheet2 = book.add_sheet("Interpolated Mid")
    sheet3 = book.add_sheet("Interpolated Fine")
    for i in range(pmc.shape[0]):
        for j in range(pmc.shape[1]):
            sheet2.write(i, j, pmc[i,j])
    for i in range(pfc.shape[0]):
        for j in range(pfc.shape[1]):    
            sheet3.write(i, j, pfc[i,j])

    if ('Solution' in runopt) | ('Uncertainty' in runopt):
        # global norms

        errc = pmc - pc
        errf = pfc - pmc

        if 'figures' in log_flag:
            plotmain(popt,xc,zc,errc,'Difference Between Mid and Coarse')
            plotmain(popt,xc,zc,errf,'Difference Between Fine and Mid')

        sheet4 = book.add_sheet("Mid - Coarse")
        sheet5 = book.add_sheet("Fine - Mid")
        for i in range(errc.shape[0]):
            for j in range(errc.shape[1]):
                sheet4.write(i, j, errc[i,j])
                sheet5.write(i, j, errf[i,j])

        # calculate convergence accross entire mesh
        if ustruc:                
            con_p = [con_p.append(calc_con(errc[i],errf[i],rc,rf))
                     for i in range(xc.shape[0])]
        else:
            if rf == rc:
                con_p = np.log(abs(errf)/abs(errc))/np.log(rf)
                con_p = abs(con_p)
                NanLoc = np.isnan(con_p)
                con_p[NanLoc] = 0. # force nans to zero
            else:
                con_p = matcon(errc,errf,rc,rf)

    if 'Solution' in runopt:
        if 'figures' in log_flag:
            # plot values equal to and below 4
            ccon_p = con_p.copy()
            ccon_p[ccon_p > 4] = 4
            plotmain(popt,xc,zc,np.array(ccon_p),'Order of Accuracy')

        sheet6 = book.add_sheet('Order of Accuracy')
        for i in range(con_p.shape[0]):
            for j in range(con_p.shape[1]):
                sheet6.write(i, j, con_p[i,j])

        # reduce extreme convergence values to 10.
        con_p[con_p > 10.] = 10.

    if not ustruc:
        # calculate relative error
        if runopt == 'Code':
            p_exact = (np.multiply(np.power(rf,con_p),pfc) - pmc) / \
                       (np.power(rf,con_p) - 1.)
            relative_err = abs((p_exact - pfc) / (p_exact + 1e-12))
            if 'figures' in log_flag:
                plotmain(popt,xc,zc,np.array(relative_err),'Relative Error')
            sheet6 = book.add_sheet("Relative Error")
            for i in range(relative_err.shape[0]):
                for j in range(relative_err.shape[1]):
                    sheet6.write(i, j, relative_err[i,j])
        # calculate GCI
        if 'Uncertainty' in runopt:
            Gcon_p = con_p.copy()
            Gcon_p[Gcon_p < 1.] = 1. # assume order of accuracy of at least one
            GCI = np.matrix([[abs(GCI_fine(pfc[i,j],pmc[i,j],rf,Gcon_p[i,j],fs)[0])
                              for j in range(pmc.shape[1])
                              ]for i in range(pmc.shape[0])])
            p_up = np.multiply(pfc,(1. + GCI))
            p_down = np.multiply(pfc,(1. - GCI))
            gci_error = np.multiply(pfc,GCI)
            sheet7 = book.add_sheet("Upper band error")
            sheet8 = book.add_sheet("Lower band error")
            sheet9 = book.add_sheet("Order of Accuracy")
            for i in range(p_up.shape[0]):
                for j in range(p_up.shape[1]):
                    sheet7.write(i, j, p_up[i,j])
                    sheet8.write(i, j, p_down[i,j])
                    sheet9.write(i, j, con_p[i,j])
            if 'figures' in log_flag:
                plotmain(popt,xc,zc,gci_error,
                        'Absolute error based on Roache GCI')
                plotmain(popt,xc,zc,p_up,
                        'Upper band of calculation error based on Roache GCI')
                plotmain(popt,xc,zc,p_down,
                        'Lower band of calculation error based on Roache GCI')
    else:
        pass
        if 'Solution' in runopt:
            sheet6 = book.add_sheet("Order of Accuracy")
            for i in range(con_p.shape[0]):
                for j in range(con_p.shape[1]):
                    sheet6.write(i, j, con_p[i,j])

    # norms global

    def verot(nm,fmt,L1c,L2c,Linfc,L1f,L2f,Linff):
        txt.insert(tk.END,32*'='+' \n',(fmt, "a"))
        txt.insert(tk.END,nm + ' Domain Norms \n',(fmt, "b"))
        eo = 'L1P1 = %(L1c)E \n' \
             'L2P1 = %(L2c)E \n' \
             'LinfP1 = %(Linfc)E \n' \
             'L1P2 = %(L1f)E \n' \
             'L2P2 = %(L2f)E \n' \
             'LinfP2 = %(Linff)E' % locals()
        txt.insert(tk.END,eo+'\n',(fmt, "a"))
      
    def conot(nm,L1_con,L2_con,L_inf_con,*args):
        eo = '%(nm)sL1 convergence: %(L1_con)0.4f \n' \
             '%(nm)sL2 convergence: %(L2_con)0.4f \n' \
             '%(nm)sL_inf convergence: %(L_inf_con)0.4f' % locals()
        try:
            eo += '\n' + nm + 'Median convergence: '+ str('%0.4f' %args[0]) \
                + '\n' + nm + 'Average convergence: '+ str('%0.4f' %args[1])
        except:
            pass
        return eo

    def domcals(nm,fmt,L1c,L2c,Linfc,L1f,L2f,Linff,med_con,ave_con):
        if 'Long log' in log_flag:
            # OP norms
            if ('Solution' in runopt) | ('Uncertainty' in runopt):
                txt.insert(tk.END,32*'='+' \n',(fmt, "a"))
                txt.insert(tk.END,nm + ' Domain Norms \n',(fmt, "b"))
                eo = 'L1c = %(L1c)0.4e \n' \
                     'L2c = %(L2c)0.4e \n' \
                     'Linfc = %(Linfc)0.4e \n' \
                     'L1f = %(L1f)0.4e \n' \
                     'L2f = %(L2f)0.4e \n' \
                     'Linff = %(Linff)0.4e' % locals()
                txt.insert(tk.END,eo+'\n',(fmt, "a"))
        if 'Solution' in runopt:
            # global convergence
            L1_con = calc_con(L1c,L1f,rc,rf)
            L2_con = calc_con(L2c,L2f,rc,rf)
            L_inf_con = calc_con(Linfc,Linff,rc,rf)
            txt.insert(tk.END,24*'-'+' \n',(fmt, "a"))
            txt.insert(tk.END,'Order of Accuracy \n',(fmt, "b"))
            eo = conot('',L1_con,L2_con,L_inf_con,med_con,ave_con)
            txt.insert(tk.END,eo+'\n',(fmt, "a"))

    def vdomcals(nm,fmt,L1_ec,L2_ec,Linf_ec,L1c,L2c,Linfc,L1f,L2f,Linff):
        if 'Long log' in log_flag:
            # OP norms
            eo = verot(nm,fmt,L1c,L2c,Linfc,L1f,L2f,Linff)
        # global convergence
        def obcon(mt,L1_con,L2_con,L_inf_con):
            txt.insert(tk.END,24*'-'+' \n',(fmt, "a"))
            txt.insert(tk.END,'Order of Accuracy \n',(fmt, "b"))
            txt.insert(tk.END,conot(mt,L1_con,L2_con,L_inf_con)+'\n',
                       (fmt, "a"))
        
        L1_con,L2_con,L_inf_con = \
                 (np.log(w)/np.log(rf) for w in [L1c/L1f,L2c/L2f,Linfc/Linff])
        obcon('P1 ',L1_con,L2_con,L_inf_con)
        
        L1_con,L2_con,L_inf_con = \
           (np.log(w)/np.log(rc) for w in [L1_ec/L1c,L2_ec/L2c,Linf_ec/Linfc])
        obcon('P2 ',L1_con,L2_con,L_inf_con)

    if ('Solution' in runopt) | ('Uncertainty' in runopt):
        # Global Domain
        L1c,L2c,Linfc = (LA.norm(np.asarray(errc.ravel())[0],w)
                         for w in [1,2,np.inf])
        L1f,L2f,Linff = (LA.norm(errf.ravel(),w) for w in [1,2,np.inf])   
        med_con = np.median(np.asarray(con_p.ravel()))
        ave_con = np.average(np.asarray(con_p.ravel()))
        domcals('Global',"n",L1c,L2c,Linfc,L1f,L2f,Linff,med_con,ave_con)
        
        # Inner Domain
        if not ustruc:
          
            L1c,L2c,Linfc = (LA.norm(np.asarray(errc[1:-1,1:-1].ravel())[0],w)
                             for w in [1,2,np.inf])
            L1f,L2f,Linff = (LA.norm(errf[1:-1,1:-1].ravel(),w)
                             for w in [1,2,np.inf])          
            med_con = np.median(np.asarray(con_p[1:-1,1:-1].ravel()))
            ave_con = np.average(np.asarray(con_p[1:-1,1:-1].ravel()))
            domcals('Inner',"a",L1c,L2c,Linfc,L1f,L2f,Linff,med_con,ave_con)
            
            # Boundary Domain
            errc[1:-1,1:-1] = 0.
            errf[1:-1,1:-1] = 0.
            con_p[1:-1,1:-1] = 999.
            icon = np.sort(np.asarray(con_p.ravel()))
            
            L1c,L2c,Linfc = (LA.norm(np.asarray(errc.ravel())[0],w)
                             for w in [1,2,np.inf]) 
            L1f,L2f,Linff = (LA.norm(errf.ravel(),w) for w in [1,2,np.inf])
            icon = icon.ravel()
            med_con = np.median(icon[0:np.max(np.where(icon<999.))].ravel())
            ave_con = np.average(icon[0:np.max(np.where(icon<999.))].ravel())
            domcals('Boundary',"n",L1c,L2c,Linfc,L1f,L2f,Linff,med_con,ave_con)
        
    else:
      
        # Global Domain
        L1_ec,L2_ec,Linf_ec = (LA.norm(np.asarray(pc.ravel())[0],w)
                               for w in [1,2,np.inf])
        L1c,L2c,Linfc = (LA.norm(pmc.ravel(),w) for w in [1,2,np.inf])
        L1f,L2f,Linff = (LA.norm(pfc.ravel(),w) for w in [1,2,np.inf])        
        vdomcals('Global',"n",L1_ec,L2_ec,Linf_ec,L1c,L2c,Linfc,L1f,L2f,Linff)

        # Inner Domain
        if not ustruc:
          
            L1_ec,L2_ec,Linf_ec = \
                              (LA.norm(np.asarray(pc[1:-1,1:-1].ravel())[0],w)
                               for w in [1,2,np.inf])
            L1c,L2c,Linfc = (LA.norm(pmc[1:-1,1:-1].ravel(),w)
                             for w in [1,2,np.inf]) 
            L1f,L2f,Linff = (LA.norm(pfc[1:-1,1:-1].ravel(),w)
                             for w in [1,2,np.inf]) 
            vdomcals('Inner',"a",L1_ec,L2_ec,Linf_ec,L1c,L2c,Linfc,L1f,L2f,
                     Linff)
            
            # Boundary Domain
            pc[1:-1,1:-1] = 0.
            pmc[1:-1,1:-1] = 0.
            pfc[1:-1,1:-1] = 0.
          
            L1_ec,L2_ec,Linf_ec = (LA.norm(np.asarray(pc.ravel())[0],w)
                                   for w in [1,2,np.inf])
            L1c,L2c,Linfc = (LA.norm(pmc.ravel(),w) for w in [1,2,np.inf])
            L1f,L2f,Linff = (LA.norm(pfc.ravel(),w) for w in [1,2,np.inf])
            vdomcals('Boundary',"n",L1_ec,L2_ec,Linf_ec,L1c,L2c,Linfc,L1f,L2f,
                     Linff)


#######################
# Graphical Interface #
#######################
import Tkinter as tk
import ttk
import tkFont

def rmrw(self,r):
    for label in self.grid_slaves():
        if int(label.grid_info()['row']) > r:
            label.destroy()
    return self
  
def wrng(wrn,pl):
    rmrw(root,pl)
    butttons(root,pl+1,'nm')
    txt = tk.Text(relief=tk.SUNKEN)
    txt.config(width=30,height=3,background="#dd4e4c")
    txt.grid(row=pl,column=0,columnspan=2,sticky=tk.W+tk.E+tk.N+tk.S,padx=5)              
    txt.tag_config("r", background="#dd4e4c", foreground="#dd4e4c")
    txt.tag_config("e", foreground="black", justify="center")
    txt.insert('1.0',wrn, ("r", "e"))

def addmenu():
    global rw,var1, var2, var3, var4, tools, Tmen
    rmrw(root,rw-1)
    if 'Uncertainty' in var1.get():
        rw += 4
        # GCI Confidence Level
        w4 = ttk.Label(root, text="GCI Confidence Level:")
        w4.grid(row=5,column=0,sticky=tk.E,padx=5)
        var4 = tk.StringVar()
        opt4 = ttk.Combobox(root,textvariable=var4,state='readonly',
                            foreground='blue',width=32)
        opt4['values'] = ('95% [1.25] (Structured Refinement)',
                          '95% [3.00] (Unstructured Refinement)',
                          '99% [1.65] (Structured Refinement)',
                          '99% [4.00] (Unstructured Refinement)')
        opt4.grid(row=5,column=1,sticky=tk.W)
        var4.set(20*' ')     
    else:
        rw += 2
    butttons(root,rw,'nm')
    # Interpolation
    w2 = ttk.Label(root, text="Interpolation:")
    w2.grid(row=rw-2,column=0,sticky=tk.E,padx=5)
    var2 = tk.StringVar()
    opt2 = ttk.Combobox(root,textvariable=var2,state='readonly',
                        foreground='blue')
    opt2['values'] = ('Linear',
                      'Spline',
                      'Cubic')
    opt2.grid(row=rw-2,column=1,sticky=tk.W)
    var2.set(20*' ')
    # Output Options
    w3 = ttk.Label(root, text="Output Options:")
    w3.grid(row=rw-1,column=0,sticky=tk.E,padx=5)
    var3 = tk.StringVar()
    opt3 = ttk.Combobox(root,textvariable=var3,state='readonly',
                        foreground='blue')
    opt3['values'] = ('Short log',
                      'Short log and figures',
                      'Long log',
                      'Long log and figures')
    opt3.grid(row=rw-1,column=1,sticky=tk.W)
    var3.set(20*' ')
    
def addmenu0_1():
    global rw, var3_1, var4_1, spcu
    rmrw(root,3)
    rw += 2
    butttons(root,rw,'nm')
    # Generate Plots
    var3_1 = tk.StringVar()
    rad1 = ttk.Frame(root)
    rad1.grid(row=4,column=1,sticky=tk.W)
    opt3 = ttk.Radiobutton(rad1, text='Yes', variable=var3_1, value='Yes')
    opt3.pack(side=tk.LEFT) 
    opt4 = ttk.Radiobutton(rad1, text='No', variable=var3_1, value='No')
    var3_1.set('No')
    opt4.pack(side=tk.LEFT)
    w3 = ttk.Label(root, text=" Generate Plots:")
    w3.grid(row=4,column=0,sticky=tk.E,padx=5)

    # Specify Cutoff
    var4_1 = tk.StringVar()
    rad2 = ttk.Frame(root)
    rad2.grid(row=5,column=1,sticky=tk.W)
    opt3 = ttk.Radiobutton(rad2, command=addmenu2_1, text='Yes',
                           variable=var4_1, value='Yes')
    opt3.pack(side=tk.LEFT)
    opt4 = ttk.Radiobutton(rad2, command=addmenu2_1, text='No',
                           variable=var4_1, value='No')
    var4_1.set('No')
    opt4.pack(side=tk.LEFT)
    w3 = ttk.Label(root, text="Specify Cutoff:")
    w3.grid(row=5,column=0,sticky=tk.E,padx=5)
    spcu = False
    
def addmenu2_1():
    global rw, var5_1, var6_1, spcu, var4_1
    if not spcu and 'Yes' in var4_1.get() and cont:
        spcu = True
        rmrw(root,5)
        rw += 1
        butttons(root,rw,'nm')
        var5_1 = tk.StringVar()
        w3 = ttk.Label(root, text="Upper Cut (%):")
        w3.grid(row=6,column=0,sticky=tk.E,padx=5)
        opt3 = ttk.Entry(root,textvariable=var5_1,foreground='blue')
        var5_1.set('100')
        opt3.grid(row=6,column=1,sticky=tk.W)
        for label in root.grid_slaves():
            if int(label.grid_info()['row'])>6:
                label.grid_forget()
        rw += 1
        butttons(root,rw,'nm')
        var6_1 = tk.StringVar()
        w4 = ttk.Label(root, text="Lower Cut (%):")
        w4.grid(row=7,column=0,sticky=tk.E,padx=5)
        opt4 = ttk.Entry(root,textvariable=var6_1,foreground='blue')
        var6_1.set('0')
        opt4.grid(row=7,column=1,sticky=tk.W)
    if spcu and 'No' in var4_1.get() and cont:
        spcu = False
        rw += -2
        butttons(root,rw,'nm')

def addmenu3():
    global rw
    try: Verification(var2.get(),var3.get(),var1.get(),var4.get())
    except ImportError:
        wrng('\nImport Error. Check for missing dependencies',rw)
    except ArrayLengthError as e:
        wrng(e.value,rw)
    except: 
        wrng('\nFile input error',rw)
    else:
        rmrw(root,rw-1)
        rw += 1
        if 'Uncertainty' in var1.get():
            ht,wd = 15,55
        else:
            ht,wd = 25,55
        sb = ttk.Scrollbar()
        sb.config(command=txt.yview)
        txt.config(yscrollcommand=sb.set,width=wd,height=ht)
        sb.grid(row=rw-1,column=2,sticky=tk.W+tk.N+tk.S)
        txt.grid(row=rw-1,column=0,columnspan=2,sticky=tk.W+tk.E+tk.N+tk.S,
                 padx=5)
        root.resizable(tk.TRUE,tk.TRUE)
        butttons(root,rw,'en')
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(rw-1, weight=1)
        
def addmenu3_1():
    global rw,txt,var3_1,var4_1,var5_1,var6_1,Validation,ValidationPlt
    
    if var3_1.get() == 'Yes':
        ValidationPlt()

    if var4_1.get() == 'Yes':
        ht,wd = 20,55
    else: 
        ht,wd = 11,55

    sb = ttk.Scrollbar()
    txt = tk.Text(relief=tk.SUNKEN)
    sb.config(command=txt.yview)
    txt.config(yscrollcommand=sb.set,width=wd,height=ht)
    
    try:
        if len(var5_1.get()) > 0:
            Validation(float(var5_1.get())/100.,float(var6_1.get())/100.)
        else:
            Validation(0.,0.)
    except ImportError:
        wrng('\nImport Error. Check for missing dependencies',rw)
    except:
        wrng('\nFile format error during read',rw)
    else:

        if var4_1.get() == 'Yes':
            rmrw(root,7)
            rw += 1
        else:
            rmrw(root,5)
            rw += 3
        
        sb.grid(row=8,column=2,sticky=tk.W+tk.N+tk.S)
        txt.grid(row=8,column=0,columnspan=2,sticky=tk.W+tk.E+tk.N+tk.S,padx=5)

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(rw-1, weight=1)
        
        root.resizable(tk.TRUE,tk.TRUE)
        
        butttons(root,rw,'en')

def openfl():
    from tkFileDialog import askopenfilename
    global fname, rw
    rmrw(root,2)
    rw += 1
    butttons(root,rw,'nm')
    w3 = ttk.Label(root, text="Input File:")
    w3.grid(row=3,column=0,sticky=tk.E,padx=5)
    
    fltp = ('*.xlsx','*.xls','*.h5','*.csv','*.txt','*.dat')
    fname = askopenfilename(filetypes=[('Input',fltp)],
                            title='Select Run File')
    if len(fname) > 0:
        w4 = ttk.Label(root, text=fname.split('/')[-1], foreground='blue')
        w4.grid(row=3,column=1,sticky=tk.W)
        if 'Validation' in var1.get():
            addmenu0_1()
        else:
            addmenu()
    else:
        wrng('\nAn input file must be selected',rw)
    
def svtext():# write output to text file
    from tkFileDialog import asksaveasfilename
    fname = asksaveasfilename(filetypes=[('Text',('*.txt'))],
                              title='Text Output Save As')
    if fname:
        with open(fname,'w') as file:
            file.write(txt.get(1.0,tk.END))
    
def svmesh():# write meshes to spreadsheet file
    from tkFileDialog import asksaveasfilename
    fname = asksaveasfilename(filetypes=[('Workbook',('*.xls'))],
                              title='Mesh Output Save As')
    if fname:
        book.save(fname)
    
def about():# VAVUQ short description
    abo = tk.Toplevel(root,background='white')
    abo.title('About VAVUQ')
    mtxt = 'VAVUQ'
    msg = ttk.Label(abo,text=mtxt,justify='center',font='TkCaptionFont',
                    background='white')
    msg.pack()
    mtxt = 'Version: 2.4'
    msg = ttk.Label(abo,text=mtxt,justify='center',font='TkHeadingFont',
                    background='white')
    msg.pack()
    mtxt = """
    VAVUQ (Verification And Validation and Uncertainty Quantification) can
    be used as a general purpose program for verification, validation, and
    uncertainty quantification. The motivation for the creation of and 
    continued development of the program is to provide a cost effective and
    easy way to assess the quality of computational approximations. The hope
    is that the methods used in this program can be applied to fields such 
    as civil engineering where they are currently often underutilized. This
    code was created through efforts from Bombardelli's group and Dr. Bill 
    Fleenor at the University of California Davis. The creators belong to 
    the Civil & Environmental Engineering (CEE) Department, the Center for
    Watershed Sciences (CWS), and the Delta Solution Team (DST). The main 
    code development is headed by Kaveh Zamani and James E. Courtney.
    """
    msg = ttk.Label(abo,text=mtxt,background='white')
    #msg.config(bg='white',font=('times',12,'italic'))
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    abo.geometry("+%d+%d"  % (x, y))
    msg.pack()

def state():
    global var1,var2,var3,var3_1,var5_1,var6_1,clr,rw,txt
    
    for label in root.grid_slaves():
        if (int(label.grid_info()['row'])<rw and 
                int(label.grid_info()['column']) == 1):
            try:
                label['state'] = 'disabled'
            except:
                pass
    
    if clr: # Clear event
        clr = False
        openfl()
    else:
        if 'Validation' in var1.get():
            if not fname:
                openfl()
            elif len(var3_1.get()) == 0:
                addmenu0_1()
            elif 'Yes' in var4_1.get():
                if len(var5_1.get()) > 0:
                    if var5_1.get() > var6_1.get():
                        butttons(root,rw,'lb')
                        root.after(10,addmenu3_1)
                    else:
                        wrng('\nUpper cut must be greater than the lower',8)
                else:
                    addmenu2_1()
            else:
                rmrw(root,rw)
                butttons(root,rw,'lb')
                root.after(10,addmenu3_1)
        else: # Verification
            if not fname:
                openfl()
            else:
                txt = tk.Text(relief=tk.SUNKEN)
                rmrw(root,rw)
                butttons(root,rw,'lb')
                root.after(10,addmenu3)  
    
def stpr():
    root.destroy()
    
def clear():
    global clr,var2,var3,var3_1,var4_1,var5_1,var6_1,rw,cont
    opt1['state'] = 'readonly'
    clr = True
    rmrw(root,2)
    var2,var3,var3_1,var4_1,var5_1,var6_1 = (tk.StringVar() for _ in xrange(6))
    root.resizable(tk.FALSE, tk.FALSE)
    butttons(root,3,'nm')
    rw = 3
    for i in range(7):
        root.rowconfigure(i, weight=0)
    root.winfo_toplevel().wm_geometry("")
    cont = True
    if specifar:
        specifar.set(0)
    for i in range(10):
        root.columnconfigure(i, weight=0)
        root.rowconfigure(i, weight=0)

def szgr(self,rw):
    self.sz = ttk.Sizegrip()
    self.sz.grid(row=rw,column=2,sticky=tk.SE)
    
def butttons(self,rw,o):
    global cont
    style = ttk.Style()
    style.configure("E.TButton", width = 7)
    style.map("E.TButton",
        foreground=[('!active', 'black'), ('pressed', 'red'),
                    ('active', 'blue')],
        background=[('!active', '#71ee6d'),('pressed', 'disabled', 'black'),
                    ('active', '#71ee6d')]
             )
    style.configure("C.TButton", width = 7)
    style.map("C.TButton",
        foreground=[('!active', 'black'),('pressed', 'red'),
                    ('active', 'blue')],
        background=[('!active', '#b8fff3'),('pressed', 'disabled', 'black'),
                    ('active', '#b8fff3')]
             )
    style.configure("X.TButton", width = 7)
    style.map("X.TButton",
        foreground=[('!active', 'black'),('pressed', 'red'),
                    ('active', 'blue')],
        background=[('!active', '#e2664c'),('pressed', 'disabled', 'black'),
                    ('active', '#e2664c')]
             )
    rmrw(self,rw-1)
    ttk.Button(self, command=clear, text='Clear',
               style="C.TButton").grid(row=rw,column=1,sticky=tk.W,padx=5,
                                       pady=5)
    ttk.Button(self, command=stpr, text='Exit',
               style="X.TButton").grid(row=rw,column=0,sticky=tk.E)   
    if o == 'lb':
        ttk.Label(self,
                  text="Processing...").grid(row=rw,column=1,sticky=tk.E,
                                             padx=5)
    elif o == 'nm':        
        ebutton = ttk.Button(self, command=state, text='Enter', 
                             style="E.TButton")
        ebutton.grid(row=rw,column=1,sticky=tk.E,padx=5)
    elif o == 'en':
        ttk.Button(self, command=state, text='Enter',
                   state=tk.DISABLED).grid(row=rw,column=1,sticky=tk.E,padx=5)
        szgr(self,rw)
        cont = False

def bug():
    def cbug(event):
        import webbrowser
        import warnings
        with warnings.catch_warnings():
            site = 'https://github.com/VAVUQ/VAVUQ'
            webbrowser.open_new(site)
            abo.destroy()
    abo = tk.Toplevel(root)
    abo.title('Contribute/Report Bug...')
    mtxt = 'Please report any bugs or contribute to VAVUQ by visiting the' \
           ' following repository:'
    msg = tk.Message(abo,text=mtxt, width=250)
    msg.config(font=('times',12))
    link = ttk.Label(abo, text="VAVUQ Repository",foreground="blue",
                     font=('times',12,'italic'), cursor="hand2")
    msg.pack()
    link.pack()
    link.bind("<Button-1>", cbug)
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    abo.geometry("+%d+%d"  % (x, y))

def doc():
    def cdoc(event):
        import webbrowser
        import warnings
        with warnings.catch_warnings():
            site = 'http://vavuq.org'
            webbrowser.open_new(site)
            abo.destroy()
    abo = tk.Toplevel(root)
    abo.title('Documentation')
    mtxt = 'Visit the following web page for the VAVUQ documentation:'
    msg = tk.Message(abo,text=mtxt, width=250)
    msg.config(font=('times',12))
    link = ttk.Label(abo, text="Vavuq.org",foreground="blue",
                     font=('times',12,'italic'), cursor="hand2")
    msg.pack()
    link.pack()
    link.bind("<Button-1>", cdoc)
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    abo.geometry("+%d+%d"  % (x, y))

def copt():
    global rw,varc1,varc2,varc3,varc4
    if specifar.get() and cont:
        rmrw(root,rw-1)
        varc1,varc2,varc3,varc4 = (tk.DoubleVar() for _ in xrange(4))
        opt = ttk.Label(root, text="X min | X max:")
        opt.grid(row=rw,column=0,sticky=tk.E,padx=5)
        cent = ttk.Frame(root)
        cent.grid(row=rw,column=1,sticky=tk.W)
        op1 = ttk.Entry(cent,textvariable=varc1,width=11,foreground='blue')
        op1.pack(side=tk.LEFT)
        op2 = ttk.Entry(cent,textvariable=varc2,width=11,foreground='blue')
        op2.pack(side=tk.LEFT)
        opt = ttk.Label(root, text="Y min | Y max:")
        opt.grid(row=rw+1,column=0,sticky=tk.E,padx=5)
        cent2 = ttk.Frame(root)
        cent2.grid(row=rw+1,column=1,sticky=tk.W)
        op3 = ttk.Entry(cent2,textvariable=varc3,width=11,foreground='blue')
        op3.pack(side=tk.LEFT)
        op4 = ttk.Entry(cent2,textvariable=varc4,width=11,foreground='blue')
        op4.pack(side=tk.LEFT)
        rw += 2
        butttons(root,rw,'nm')
    elif cont:
        rmrw(root,rw-3)
        rw += -2
        butttons(root,rw,'nm')

global cont

cont = True

root = tk.Tk()

root.title("VAVUQ")

menubar = ttk.Frame(root)
menubar.grid(row=0,column=0,columnspan=2,sticky=tk.W)

Fmen = tk.Menubutton(menubar, text='File', underline=0)
Fmen.pack(side=tk.LEFT)
fle = tk.Menu(Fmen,tearoff=0)
fle.add_command(command=clear,label='Clear', underline=0)
fle.add_separator()
fle.add_command(command=svtext, label='Save Output Text As..', underline=12)
fle.add_command(command=svmesh, label='Save Output Meshes As..', underline=12)
fle.add_separator()
fle.add_command(command=stpr, label='Quit', underline=0)
Fmen.config(menu=fle)

Pmen = tk.Menubutton(menubar, text='Plotting', underline=0)
Pmen.pack(side=tk.LEFT)
plm = tk.Menu(Pmen, tearoff=0)
Pmen.config(menu=plm)
#Verification plotting submenu
vasub = tk.Menu(Pmen, tearoff=0)

Surf,Scat,Cont,Imag = (tk.BooleanVar() for _ in xrange(4))
Surf.set(True)
vasub.add_checkbutton(label='Surface Plot', onvalue=1, offvalue=0,
                      variable=Surf)
vasub.add_checkbutton(label='Scatter Plot', onvalue=1, offvalue=0,
                      variable=Scat)
vasub.add_checkbutton(label='Contour Plot', onvalue=1, offvalue=0,
                      variable=Cont)
vasub.add_checkbutton(label='Image Plot', onvalue=1, offvalue=0,
                      variable=Imag)

plm.add_cascade(label='Verification', menu=vasub, underline=0)

# Validation plotting submenu
vesub = tk.Menu(Pmen, tearoff=0)

Obvc,Obwc,Obmc = (tk.BooleanVar() for _ in xrange(3))
Obvc.set(True)
vesub.add_checkbutton(label='Observed V.S. Calc Plot', onvalue=1, offvalue=0,
                      variable=Obvc)
vesub.add_checkbutton(label='Observed W/ Calc Plot', onvalue=1, offvalue=0,
                      variable=Obwc)
vesub.add_checkbutton(label='Observed - Calc Plot', onvalue=1, offvalue=0,
                      variable=Obmc)

plm.add_cascade(label='Validation', menu=vesub, underline=0)
    
specifar = tk.BooleanVar()

Tmen = tk.Menubutton(menubar, text='Tools', underline=0)
Tmen.pack(side=tk.LEFT)
tools = tk.Menu(Tmen,tearoff=0)
tools.add_checkbutton(command=copt, label='Cut Mesh', onvalue=1, offvalue=0,
                      variable=specifar)
Tmen.config(menu=tools)

Hmen = tk.Menubutton(menubar, text='Help', underline=0)
Hmen.pack(side=tk.LEFT)
hlp = tk.Menu(Hmen,tearoff=0)
hlp.add_command(command=doc, label='Documentation', underline=0)
hlp.add_command(command=bug, label='Contribute/Report Bug...', underline=0)
hlp.add_command(command=about, label='About VAVUQ', underline=0)
Hmen.config(menu=hlp)

root.sp = ttk.Separator(orient=tk.HORIZONTAL)
root.sp.grid(row=1,column=0,columnspan=3, sticky=tk.EW)

rw = 2

var1,var2,var3,var4,var3_1,var4_1,var5_1,var6_1 = (tk.StringVar()
                                                   for _ in xrange(8))
clr,fname,fname = False, False, False
w1 = ttk.Label(root, text=" Run Option:")
w1.grid(row=rw,column=0,sticky=tk.E,padx=5)

def kpr(evt):
    if cont: state()

opt1 = ttk.Combobox(root,textvariable=var1,state='readonly',foreground='blue',
                    width=34)
opt1['values'] = ('Validation',
                  'Code Verification (MMS, MES, or DM)',
                  'Solution Verification',
                  'Uncertainty Quantification')

opt1.bind('<<ComboboxSelected>>', kpr)
opt1.grid(row=rw,column=1,sticky=tk.W)
var1.set(55*' ')

rw += 1
butttons(root,rw,'nm')

root.withdraw()
root.update_idletasks()
x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2 - 250
root.geometry("+%d+%d"  % (x, y))
root.deiconify()

root.resizable(tk.FALSE, tk.FALSE)

root.bind("<Return>", kpr)
root.focus_set()

root.mainloop()
