import sys
sys.path.append('../')
from functions import *  
import os
from time import perf_counter

if __name__ == '__main__':

    # read grid -------
    xgrid, ygrid, zgrid = read_grid()
    nx,ny,nz = len(xgrid), len(ygrid), len(zgrid)
    print('nx, ny, nz =', nx, ny, nz)

    # read parallelization info -----
    npe, iproc, jproc, nvar = read_parallel_info()                                 
    print('npe, iproc, jproc, nvar = ', npe, iproc, jproc, nvar)

    # read EBM info
    t_EBM, radius, Ur = read_EBM()

    # list output files
    files = sorted(glob.glob('./output/out*dat'))
    nout = len(files)

    # compile downsampled distribution
    for nt in range(nout):

        t, uu = read_output_location(files[nt],0,0,0,nvar,nx,ny,nz)
        print('t = {:.3f}'.format(t))

        if os.path.exists('./output/Babs_ds_{:03d}.npy'.format(nt)):
            print('Downsampled File exists, skip...')
            rho_ds = np.load('./output/rho_ds_{:03d}.npy'.format(nt))
            vx_ds = np.load('./output/vx_ds_{:03d}.npy'.format(nt))
            vy_ds = np.load('./output/vy_ds_{:03d}.npy'.format(nt))
            vz_ds = np.load('./output/vz_ds_{:03d}.npy'.format(nt))
            Bx_ds = np.load('./output/Bx_ds_{:03d}.npy'.format(nt))
            By_ds = np.load('./output/By_ds_{:03d}.npy'.format(nt))
            Bz_ds = np.load('./output/Bz_ds_{:03d}.npy'.format(nt))
            Babs_ds = np.load('./output/Babs_ds_{:03d}.npy'.format(nt))
            vabs_ds = np.load('./output/vabs_ds_{:03d}.npy'.format(nt))
            p_ds = np.load('./output/p_ds_{:03d}.npy'.format(nt))
        else:
            # read file
            print('Begin reading file...')
            time0 = perf_counter()

            tmp, rho = read_output_onevariable(files[nt],0,nvar,nx,ny,nz)
            tmp, vx = read_output_onevariable(files[nt],1,nvar,nx,ny,nz)
            tmp, vy = read_output_onevariable(files[nt],2,nvar,nx,ny,nz)
            tmp, vz = read_output_onevariable(files[nt],3,nvar,nx,ny,nz)
            tmp, Bx = read_output_onevariable(files[nt],4,nvar,nx,ny,nz)
            tmp, By = read_output_onevariable(files[nt],5,nvar,nx,ny,nz)
            tmp, Bz = read_output_onevariable(files[nt],6,nvar,nx,ny,nz)
            tmp, p = read_output_onevariable(files[nt],7,nvar,nx,ny,nz)

            print("Number of Points: {}".format(len(Bx.flatten())))

            time1 = perf_counter()
            print('Time spent: {:.3E} sec'.format(time1-time0))

            # downsample
            print('Begin downsample...')
            time0 = perf_counter()

            rho_ds = downsample(rho,xgrid,ygrid,zgrid,factor=2)[0]
            vx_ds = downsample(vx,xgrid,ygrid,zgrid,factor=2)[0]
            vy_ds = downsample(vy,xgrid,ygrid,zgrid,factor=2)[0]
            vz_ds = downsample(vz,xgrid,ygrid,zgrid,factor=2)[0]
            Bx_ds = downsample(Bx,xgrid,ygrid,zgrid,factor=2)[0]
            By_ds = downsample(By,xgrid,ygrid,zgrid,factor=2)[0]
            Bz_ds = downsample(Bz,xgrid,ygrid,zgrid,factor=2)[0]
            p_ds = downsample(p,xgrid,ygrid,zgrid,factor=2)[0]

            Babs_ds = np.sqrt(Bx_ds**2 + By_ds**2 + Bz_ds**2)
            vabs_ds = np.sqrt(vx_ds**2 + vy_ds**2 + vz_ds**2)


            time1 = perf_counter()
            print('Time spent: %.2f sec'%(time1-time0))


            np.save('./output/rho_ds_{:03d}.npy'.format(nt),rho_ds)
            np.save('./output/vx_ds_{:03d}.npy'.format(nt),vx_ds)
            np.save('./output/vy_ds_{:03d}.npy'.format(nt),vy_ds)
            np.save('./output/vz_ds_{:03d}.npy'.format(nt),vz_ds)
            np.save('./output/Bx_ds_{:03d}.npy'.format(nt),Bx_ds)
            np.save('./output/By_ds_{:03d}.npy'.format(nt),By_ds)
            np.save('./output/Bz_ds_{:03d}.npy'.format(nt),Bz_ds)
            np.save('./output/p_ds_{:03d}.npy'.format(nt),p_ds)
            np.save('./output/vabs_ds_{:03d}.npy'.format(nt),vabs_ds)
            np.save('./output/Babs_ds_{:03d}.npy'.format(nt),Babs_ds)

        # var dict
        var_dict = {
            'rho':rho_ds,
            'vx':vx_ds,
            'vy':vy_ds,
            'vz':vz_ds,
            'Bx':Bx_ds,
            'By':By_ds,
            'Bz':Bz_ds,
            'p':p_ds,
            'Babs':Babs_ds,
            'vabs':vabs_ds
        }

        print('Begin calculate distribution...')
        time0 = perf_counter()

        # print("Number of Points: {}".format(len(Babs_ds.flatten())))


        # ====== create figures ====== #
        os.makedirs('./figure',exist_ok=True)


        var_list = ['rho','vx','vy','vz','Bx','By','Bz','p','Babs','vabs']

        for var in var_list:
            os.makedirs('./figure/{}'.format(var),exist_ok=True)

            # calculate the Gaussian fit
            mean, std = np.mean(var_dict[var].flatten()), np.std(var_dict[var].flatten())
            x = np.linspace(mean - 5*std, mean + 5*std, 1000)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)

            fig, axes = plt.subplots(1,2,figsize=(8,4), layout='constrained')
            
            ax = axes[0]
            plt.sca(ax)
            plt.hist(var_dict[var].flatten(),bins=200,range=(mean-5*std,mean+5*std), density=True, alpha = 0.2)
            plt.plot(x,y,color='C1')
            plt.axvline(mean,color='k',ls='--')
            ax.set_xlabel(var,fontsize=14)
            ax.set_ylabel(r'number of points',fontsize=12)

            ax = axes[1]
            plt.sca(ax)
            plt.hist(var_dict[var].flatten(),bins=200,range=(mean-5*std,mean+5*std), density=True, alpha = 0.2)
            plt.plot(x,y,color='C1')
            plt.axvline(mean,color='k',ls='--')
            ax.set_yscale('log',base=10)
            ax.set_xlabel(var,fontsize=14)
            ax.set_ylabel(r'number of points',fontsize=12)

            fig.suptitle(r'$var= %s, t = %.3f$' %(var,t))
            fig.savefig('./figure/{}/distribution_{:03d}.png'.format(var,nt))

            plt.close()


        # ====== Time spent ====== #
        time1 = perf_counter()
        print('Time spent: {:.2F} sec'.format(time1-time0))

