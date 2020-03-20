import numpy as np

maxIter = 1000
Re = 10.0
nx, ny = 420, 180
ly = ny-1
cx, cy, r = nx//4, ny//2, ny//9
uLB = 0.04
nulb = uLB*r/Re
omega = 1 / (3*nulb+0.5)

v = np.array([
    [1,1], [1,0], [1,-1], [0,1], [0,0],
    [0,-1], [-1,1], [-1,0], [-1,-1]
])
t = np.array([
    1/36,1/9,1/36,1/9,4/9,1/9,1/36,1/9,1/36
])
col1 = np.array([0,1,2])
col2 = np.array([3,4,5])
col3 = np.array([6,7,8])

if __name__ == "__main__":

    def macroscopic(fin):
        rho = np.sum(fin, axis=0)
        u = np.zeros((2,nx,ny))
        for i in range(9):
            u[0,:,:] += v[i,0] * fin[i,:,:]
            u[1,:,:] += v[i,1] * fin[i,:,:]
        u /= rho
        return rho, u

    def equilibrium(rho, u):
        usqr = 3/2 * (u[0]**2 + u[1]**2)
        feq = np.zeros((9,nx,ny))
        for i in range(9):
            cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
            feq[i,:,:] = rho*t[i] * (1+cu+0.5*cu**2 - usqr)
        return feq

    def obstacle_fun(x, y):
        return ((x-cx)**2 + (y-cy)**2) < r**2

    obstacle = np.fromfunction(obstacle_fun, (nx,ny))

    def inivel(d, x, y):
        return (1-d) * uLB * (1 + 1e-4*np.sin(y/ly * 2 * np.pi))

    vel = np.fromfunction(inivel, (2,nx,ny))

    fin = equilibrium(1, vel)

    for time in range(maxIter):
        fin[col3,-1,:] = fin[col3,-2,:]
        rho, u = macroscopic(fin)
        u[:,0,:] = vel[:,0,:]
        rho[0,:] = 1/(1-u[0,0,:]) * (
            np.sum(fin[col2,0,:], axis=0) + \
            2 * np.sum(fin[col3,0,:], axis=0)
        )
        
        feq = equilibrium(rho, u)
        fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]
        
        fout = fin - omega * (fin - feq)
        
        for i in range(9):
            fout[i, obstacle] = fin[8-i, obstacle]
            
        for i in range(9):
            fin[i,:,:] = np.roll(
                np.roll(
                    fout[i,:,:], v[i,0], axis=0
                ),
                v[i,1], axis=1
            )
            
        if (time%999 == 0):
            plt.clf()
            plt.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
            plt.savefig("vel,{0:03d}.png".format(time//100))
