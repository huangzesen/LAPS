import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import glob
import sys
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from numba import njit
from math import floor,sqrt


def read_parallel_info():
    # read parallel information ----------------
    file_prl = open('./output/parallel_info.dat','rb')

    skip = struct.unpack("f",file_prl.read(4))

    npe = int((struct.unpack("f",file_prl.read(4)))[0])
    iproc = int((struct.unpack("f",file_prl.read(4)))[0])
    jproc = int((struct.unpack("f",file_prl.read(4)))[0])
    nvar = int((struct.unpack("f",file_prl.read(4)))[0])

    skip = struct.unpack("f",file_prl.read(4))

    file_prl.close()
    return npe, iproc, jproc, nvar


def read_EBM():
    #read the EBM_info.dat
    #which includes time,radius,Ur

    file_EBM = np.array(np.loadtxt('./output/EBM_info.dat'))

    if len(file_EBM.shape)==1:
        file_EBM = np.reshape(file_EBM,(int(len(file_EBM)/3),3))

    t_EBM = file_EBM[:,0]
    radius = file_EBM[:,1]
    Ur_EBM = file_EBM[:,2]

    return t_EBM,radius,Ur_EBM

def read_grid():
    # read nx, ny, nz and grid-------------------------
    file_grid = open('./output/grid.dat', 'rb')

    skip = struct.unpack("f",file_grid.read(4))

    nx = int((struct.unpack("f",file_grid.read(4)))[0])
    ny = int((struct.unpack("f",file_grid.read(4)))[0])
    nz = int((struct.unpack("f",file_grid.read(4)))[0])

    skip = struct.unpack("f",file_grid.read(4))

    xgrid = np.zeros(nx)
    ygrid = np.zeros(ny)
    zgrid = np.zeros(nz)


    skip = struct.unpack("f",file_grid.read(4))

    for i in range(nx):
        xgrid[i] = (struct.unpack("f",file_grid.read(4)))[0]
    for i in range(ny):
        ygrid[i] = (struct.unpack("f",file_grid.read(4)))[0]
    for i in range(nz):
        zgrid[i] = (struct.unpack("f",file_grid.read(4)))[0]

    skip = struct.unpack("f",file_grid.read(4))

    file_grid.close()
    return xgrid,ygrid,zgrid


def read_uu(filename,nx,ny,nz,nvar):
    file_uu = open(filename, 'rb')
    uu = np.zeros([nx,ny,nz,nvar])

    skip = (struct.unpack("f",file_uu.read(4)))[0]
    t = (struct.unpack("f",file_uu.read(4)))[0] 
    skip = (struct.unpack("f",file_uu.read(4)))[0]

    for ivar in range(nvar):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    uu[ix,iy,iz,ivar] = (struct.unpack(\
                        "d",file_uu.read(8)))[0] 

    file_uu.close()
    return t, uu


def read_output_location(filename,ix,iy,iz,nvar,nx,ny,nz):
    file = open(filename, 'rb')

    skip = (struct.unpack("f",file.read(4)))[0]
    t = (struct.unpack("f",file.read(4)))[0] 
    skip = (struct.unpack("f",file.read(4)))[0]

    offset = file.tell()

    vals = np.zeros(nvar)
    for iv in range(nvar):
        indx =  ((iv * nz + iz) * ny + iy) * nx + ix
        file.seek(indx * 8 + offset)
        vals[iv] = struct.unpack('d',file.read(8))[0]

    file.close()

    return t, vals

def read_output_slice(filename,indices,nvar,nx,ny,nz):
    indices = np.array(indices)
    dims = np.array([nx,ny,nz], dtype=int)

    axis = np.where(indices==-1)[0]
    if len(axis)!=2:
        print('In read_output_slice: two axes need to be specified!')
        return None, None

    axis_fix = np.where(indices>=0)[0][0]

    arr = np.zeros(np.append(nvar,dims[axis]))

    file = open(filename,'rb')

    skip = (struct.unpack("f",file.read(4)))[0]
    t = (struct.unpack("f",file.read(4)))[0] 
    skip = (struct.unpack("f",file.read(4)))[0]

    offset = file.tell()
    offset_now = offset

    loc = np.zeros(3,dtype=int)
    loc[axis_fix] = indices[axis_fix]
    
    for iv in range(nvar):
        for i2 in range(dims[axis[1]]):
            loc[axis[1]] = i2
            for i1 in range(dims[axis[0]]):
                loc[axis[0]] = i1 
        
                indx = ((iv * nz + loc[2]) * ny + loc[1]) * nx + loc[0]
                offset_new = indx*8 + offset

                if offset_new != offset_now:
                    file.seek(offset_new - offset_now, 1)

                arr[iv,i1,i2] = struct.unpack('d',file.read(8))[0]
                offset_now = file.tell()

    file.close()

    return t, arr

def read_output_axis(filename,indices,nvar,nx,ny,nz):
    indices = np.array(indices)
    dims = np.array([nx,ny,nz], dtype=int)

    axis = np.where(indices==-1)[0]
    if len(axis)!=1:
        print('In read_output_slice: One axis needs to be specified!')
        return None, None

    axis_fix = np.where(indices>=0)[0]

    arr = np.zeros(np.append(nvar,dims[axis]))

    file = open(filename,'rb')

    skip = (struct.unpack("f",file.read(4)))[0]
    t = (struct.unpack("f",file.read(4)))[0] 
    skip = (struct.unpack("f",file.read(4)))[0]

    offset = file.tell()
    offset_now = offset

    loc = np.zeros(3,dtype=int)
    loc[axis_fix] = indices[axis_fix]

    for iv in range(nvar):

        for i1 in range(dims[axis[0]]):
            loc[axis[0]] = i1 
            
            indx = ((iv * nz + loc[2]) * ny + loc[1]) * nx + loc[0]
            offset_new = indx*8 + offset

            if offset_new != offset_now:
                file.seek(offset_new - offset_now, 1)
                
            arr[iv,i1] = struct.unpack('d',file.read(8))[0]
            offset_now = file.tell()

    file.close()

    return t, arr


def read_output_onevariable(filename,iv,nvar,nx,ny,nz):
    arr = np.zeros([nx,ny,nz])

    file = open(filename, 'rb')

    skip = (struct.unpack("f",file.read(4)))[0]
    t = (struct.unpack("f",file.read(4)))[0] 
    skip = (struct.unpack("f",file.read(4)))[0]

    offset0 = file.tell()

    offset_now = offset0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                indx = ((iv * nz + iz) * ny + iy) * nx + ix
                offset_new = indx * 8 + offset0

                if offset_new != offset_now:
                    file.seek(offset_new - offset_now, 1)

                arr[ix,iy,iz] = struct.unpack('d',file.read(8))[0]
                offset_now = file.tell()
    
    file.close()
    return t,arr






@njit
def SF_2(arr,l=[0,0,0]):   # SF_2(l) = < [arr(x+l) - arr(x)]^2 >_x
    # Specify a vector l = (lx,ly,lz) and calculate SF2
    shape = arr.shape
    nx = shape[0]
    ny = shape[1]
    nz = shape[2]

    N = nx * ny * nz 

    lx = l[0]
    ly = l[1]
    lz = l[2]

    S2 = 0 

    for i in range(nx):
        i1 = (i + lx)%nx 
        for j in range(ny):
            j1 = (j + ly)%ny
            for k in range(nz):
                k1 = (k + lz)%nz 
                
                S2 = S2 + (arr[i1,j1,k1] - arr[i,j,k])**2


    S2 = S2/N 

    return S2



@njit(parallel=True)
def SF_2_axis(arr,axis=0):
    # note: we only calculate l_vector = (lx,0,0) or (0,ly,0) or (0,0,lz)
    # Calculating all l_vector is too expensive
    shape = arr.shape
    n_axis = shape[axis]

    l = [0,0,0]

    S2_x = np.zeros(n_axis)

    for ix in range(n_axis):
        l[axis] = ix
        S2_x[ix] = SF_2(arr,l)

    return S2_x


@njit(parallel=True)
def SF_q_mag(arr,l=[0,0,0],q=2):
    # arr = [bx,by,bz]
    # SF_q(l) = < [|B_vec(x+l) - B_vec(x)|]^q >_x
    # Specify a vector l = (lx,ly,lz) and calculate SF_q

    # shape = arr.shape
    if len(arr)!=3:
        print('In function SF_q_mag: arr should be 3*nx*ny*nz!')
        return 
    
    bx = arr[0]
    by = arr[1]
    bz = arr[2]
    shape = bx.shape
    nx = shape[0]
    ny = shape[1]
    nz = shape[2]

    N = nx * ny * nz 

    lx = l[0]
    ly = l[1]
    lz = l[2]

    S_q = 0 

    for i in range(nx):
        i1 = (i + lx)%nx 
        for j in range(ny):
            j1 = (j + ly)%ny
            for k in range(nz):
                k1 = (k + lz)%nz 
                
                S_q = S_q + (sqrt((bx[i1,j1,k1] - bx[i,j,k])**2 + \
                    (by[i1,j1,k1] - by[i,j,k])**2 + \
                    (bz[i1,j1,k1] - bz[i,j,k])**2))**q


    S_q = S_q/N 

    return S_q

@njit(parallel=True)
def SF_q_mag_axis(arr,axis=0,q=2):
    # note: we only calculate l_vector = (lx,0,0) or (0,ly,0) or (0,0,lz)
    # Calculating all l_vector is too expensive

    if len(arr)!=3:
        print('In function SF_q_mag: arr should be 3*nx*ny*nz!')
        return 

    shape = arr[0].shape
    n_axis = shape[axis]

    l = [0,0,0]

    Sq_x = np.zeros(n_axis)

    for ix in range(n_axis):
        l[axis] = ix
        Sq_x[ix] = SF_q_mag(arr,l,q)

    return Sq_x



def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

def downsample(arr,xgrid,ygrid,zgrid,factor=2):
    if not is_power_of_two(factor):
        print('In function downsample: factor must be factor of 2!!')
        return 

    nx = arr.shape[0]
    ny = arr.shape[1]
    nz = arr.shape[2]

    nx_new = int(nx/factor)
    ny_new = int(ny/factor)
    nz_new = int(nz/factor)

    ix_new = np.arange(0,nx,factor)
    iy_new = np.arange(0,ny,factor)
    iz_new = np.arange(0,nz,factor)

    arr_new = np.zeros([nx_new,ny_new,nz_new])
    for i in range(nx_new):
        for j in range(ny_new):
            for k in range(nz_new):
                arr_new[i,j,k] = arr[ix_new[i],iy_new[j],iz_new[k]]

    xgrid_new = np.zeros(nx_new)
    ygrid_new = np.zeros(ny_new)
    zgrid_new = np.zeros(nz_new)

    for i in range(nx_new):
        xgrid_new[i] = xgrid[ix_new[i]]

    for i in range(ny_new):
        ygrid_new[i] = ygrid[iy_new[i]]

    for i in range(nz_new):
        zgrid_new[i] = zgrid[iz_new[i]]

    return arr_new,xgrid_new,ygrid_new,zgrid_new


def distribution(arr,low=-1,up=1,n=10):
    # give 1d array, calculate distribution (even-spaced)

    if len(arr.shape)!=1:
        # print('In function distribution(), dimension of arr must be 1!!')
        # return 
        arr_flat = arr.flatten()
    else:
        arr_flat = np.copy(arr)

    bound = np.linspace(low,up,n+1)

    center = np.zeros(n)
    for i in range(n):
        center[i] = (bound[i+1]+bound[i])/2

    inc = bound[1]-bound[0]

    num = np.zeros(n)

    for i in range(len(arr_flat)):
        ind = floor((arr_flat[i]-low)/inc)

        if ind<0 or ind>n-1:
            continue 

        num[ind] = num[ind] + 1

    return center,num


def fit_norm(x,y):
    Y = np.log(y)
    X = x

    fit_ = np.polyfit(X,Y,2)

    sigma = np.sqrt(-1/fit_[0])

    xc = 0.5 * sigma**2 * fit_[1]

    Y0 = xc**2/sigma**2 + fit_[2]

    y0 = np.exp(Y0)

    return y0,xc,sigma




def rotate_to_magnetic_field_coordinate(arr,xgrid,ygrid,angle):
    # assume the magnetic field is in x-y plane and 
    # has an angle (radian) w.r.t. x-axis

    # arr should be either a [nvar,nx,ny,nz] array
    # or a [nx,ny,nz] array

    shape = arr.shape

    if len(shape)==4:
        n_var = shape[0]
        nx = shape[1]
        ny = shape[2]
        nz = shape[3]
    else:
        n_var = 1
        nx = shape[0]
        ny = shape[1]
        nz = shape[2]

    if nx!=len(xgrid) or ny!=len(ygrid):
        print('In function rotate_to_magnetic_field_coordinate():' + 
              'shape of arr does not match shape of xgrid or ygrid!')
    
        # sys.exit()

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    arr_new = np.zeros(arr.shape)

    for i in range(nx):
        for j in range(ny):
            x_new = xgrid[i]
            y_new = ygrid[j]

            x_origin = x_new * np.cos(angle) + y_new * np.sin(angle)
            y_origin = -x_new * np.sin(angle) + y_new * np.cos(angle)


            ind_x = floor(x_origin/dx)
            ind_y = floor(y_origin/dy)

            x_dist0 = x_origin - ind_x * dx 
            y_dist0 = y_origin - ind_y * dy
            x_dist1 = dx - x_dist0
            y_dist1 = dy - y_dist0

            if ind_x<0:
                ind_x = ind_x + nx 
            elif ind_x>=nx:
                ind_x = ind_x%nx 
            if ind_y<0:
                ind_y = ind_y + ny 
            elif ind_y>=ny:
                ind_y = ind_y%ny

            ind_x1 = (ind_x+1)%nx 
            ind_y1 = (ind_y+1)%ny

            if len(shape)==4:
                # for k in range(nz):
                #     for iv in range(n_var):
                #         arr_new[iv,i,j,k] = (arr[iv,ind_x,ind_y,k]*x_dist1*y_dist1 + 
                #             arr[iv,ind_x1,ind_y,k]*x_dist0*y_dist1 +
                #             arr[iv,ind_x,ind_y1,k]*x_dist1*y_dist0 +
                #             arr[iv,ind_x1,ind_y1,k]*x_dist0*y_dist0)/dx/dy
                arr_new[:,i,j,:] = (arr[:,ind_x,ind_y,:]*x_dist1*y_dist1 + 
                        arr[:,ind_x1,ind_y,:]*x_dist0*y_dist1 +
                        arr[:,ind_x,ind_y1,:]*x_dist1*y_dist0 +
                        arr[:,ind_x1,ind_y1,:]*x_dist0*y_dist0)/dx/dy
            else:
                # for k in range(nz):
                #     arr_new[i,j,k] = (arr[ind_x,ind_y,k]*x_dist1*y_dist1 + 
                #         arr[ind_x1,ind_y,k]*x_dist0*y_dist1 +
                #         arr[ind_x,ind_y1,k]*x_dist1*y_dist0 +
                #         arr[ind_x1,ind_y1,k]*x_dist0*y_dist0)/dx/dy
                arr_new[i,j,:] = (arr[ind_x,ind_y,:]*x_dist1*y_dist1 + 
                        arr[ind_x1,ind_y,:]*x_dist0*y_dist1 +
                        arr[ind_x,ind_y1,:]*x_dist1*y_dist0 +
                        arr[ind_x1,ind_y1,:]*x_dist0*y_dist0)/dx/dy


    return arr_new