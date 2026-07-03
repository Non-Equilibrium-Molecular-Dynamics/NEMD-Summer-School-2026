import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import DBSCAN

from view_domain import xyz_num_frames, read_frame

def wrap_pbc(points, Lx, Ly):
    """Wrap x,y into minimum image convention box [0,L)."""
    p = points.copy()
    p[:, 0] = p[:, 0] % Lx
    p[:, 1] = p[:, 1] % Ly
    return p

def liquid_vapour_dbscan(points, eps=1.5, min_samples=10, Lx=None, Ly=None):
    """
    Identify liquid cluster in slab geometry (PBC in x,y, free in z).
    
    Returns:
        labels, liquid_mask
    """

    if Lx is not None and Ly is not None:
        points = wrap_pbc(points, Lx, Ly)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # remove noise label (-1)
    valid = labels >= 0
    if not np.any(valid):
        return labels, np.zeros(len(points), dtype=bool)

    # largest cluster = liquid
    counts = np.bincount(labels[valid])
    liquid_label = np.argmax(counts)

    liquid_mask = labels == liquid_label

    return labels, liquid_mask


#Copied from pytim surface code

def _compute_q_vectors(box, normal=2, alpha=0.5):

    """ 
        Compute the q-vectors compatible with the current box dimensions.
        Inputs:
        box       : List of domain dimensions [Lx, Ly, Lz]
        normal    : Normal to surface x=0, y=1 and z=2
        alpha     : Molecular scale cutoff, default 2.0
        Outputs:
        q_vectors : two 2D arrays forming the grid of q-values, similar
                    to a meshgrid
        mode_shape: Number of modes
        Qxy       : array of the different q-vectors
        Q         : squared module of Qxy with the first element missing
                    (no Q = 0.0)
    """

    box = np.roll(box, 2 - normal)
    nmax = list(map(int, np.ceil(box[0:2] / alpha)))
    q_vectors = np.mgrid[0:nmax[0], 0:nmax[1]] * 1.0
    q_vectors[0] *= 2. * np.pi / box[0]
    q_vectors[1] *= 2. * np.pi / box[1]
    modes_shape = q_vectors[0].shape
    qx = q_vectors[0][:, 0]
    qy = q_vectors[1][0]
    Qx = np.repeat(qx, len(qy))
    Qy = np.tile(qy, len(qx))
    Qxy = np.vstack((Qx, Qy)).T
    Q = np.sqrt(np.sum(Qxy * Qxy, axis=1)[1:])

    return q_vectors, modes_shape, Qxy, Q

def get_surface_modes(points, Qxy, modes_shape, Q=None, constraint=False):

    if np.any(Q == None):
        Q = np.sqrt(np.sum(Qxy * Qxy, axis=1)[1:])
    #QR = np.dot(Qxy, points[:, 0:2].T).T


    QR = np.zeros([points.shape[0],Qxy.shape[0]])

    for i in range(QR.shape[0]):
        for j in range(QR.shape[1]):
            QR[i,j] = points[i,0]*Qxy[j,0] + points[i,1]*Qxy[j,1]

    ph = (np.cos(QR) + 1.j * np.sin(QR))[:, 1:]

    z = points[:, 2]
    az = np.mean(z)
    z = z - az
    A = (ph / Q)

    pinv_ph = np.linalg.pinv(A)

    if constraint != False:
        for i in range(A.shape[0]):
            A[i,:] += constraint*Q

    s = np.zeros(modes_shape[0]*modes_shape[1]-1, dtype=complex)
    for i in range(s.shape[0]):
        s[i] = np.dot(pinv_ph[i,:], z[:])/Q[i] 

    out = np.append(az + 0.j, s)
    modes = out.reshape(modes_shape)

    # return the surface modes reshaped into an array
    return modes

def get_phase(point, q_vectors, derivative=None):
    dotp = q_vectors[0] * point[0] + q_vectors[1] * point[1]
    if derivative is None:
        return np.cos(dotp) + 1.j * np.sin(dotp)
    elif derivative == "x":
        #dfdx
        return (-q_vectors[0]*np.sin(dotp) 
                + 1.j*q_vectors[0]*np.cos(dotp))
    elif derivative == "y":
        #dfdy
        return (-q_vectors[1]*np.sin(dotp) 
                + 1.j*q_vectors[1]*np.cos(dotp))
    elif derivative == "xy" or derivative == "yx":
        #d2fdxy
        return (-q_vectors[0]*q_vectors[1]*np.cos(dotp) 
                - 1.j*q_vectors[0]*q_vectors[1]*np.sin(dotp))
    else:
        print("Error -- derivative is unknown")

def surface_from_modes(points, q_vectors, modes, derivative=None):
    elevation = []
    for point in points:
        phase = get_phase(point, q_vectors, derivative=derivative)
        elevation.append(np.sum((phase * modes).real))
    return np.array(elevation)

def surface(points, derivative=None):
    return surface_from_modes(points, q_vectors, 
                              modes, derivative=derivative)

def get_initial_pivots(points, box, normal):
    """ 
        Defines the initial pivots as a set of 9 particles, where
        each particle is in a distinct sector formed by dividing
        the macroscopic plane into 3x3 regions.
    """
    sorted_ind = np.argsort(points[:,normal])
    sectors = np.zeros((3, 3), dtype=int)
    pivot = []
    for ind in sorted_ind:
        part = points[ind]
        nx, ny = list(map(int, 2.999 * part[0:2] / box[0:2]))
        if sectors[nx, ny] == 0:
            pivot.append(ind)
            sectors[nx, ny] = 1
        if np.sum(sectors) >= 9:
            break
    return pivot


def update_pivots(points, q_vectors, modes, pivot, normal, alpha, tau):
    """ 
        Searches for points within a distance tau from the
        interface.
    """
    pivot_pos = points[pivot]
    z_max = np.max(pivot_pos[:,normal]) + alpha * 2
    z_min = np.min(pivot_pos[:,normal]) - alpha * 2
    z = points[:, normal]
    condition = np.logical_and(z > z_min, z < z_max)
    candidates = np.argwhere(condition)[:, 0]
    dists = surface_from_modes(points[candidates], q_vectors, modes)
    dists = dists - z[candidates]
    return candidates[dists*dists < tau**2]

if __name__ == "__main__":

    TRAJ = "day2_traj.xyz"
    plotmols = True
    normal = 2    #Assume normal in z direction
    alpha = 0.5   #Minimum wavelength on surface
    tau = 0.5     #starting search criteria
    ns = 0.4      #Target density at surface
    nbins = 100   #Number of bins
    Ntries = 100  #Fitting attempts

    #Domain size (hardwired, can get from log or xyz min/max)
    Lx = 10.7268424211588
    Ly = 10.7268424211588
    Lz = 89.390353509657
    box = [Lx, Ly, Lz]
    Area = Lx * Ly

    #Create a meshgrid to plot surface
    N = 100
    x = np.linspace(-Lx/2., Lx/2., N)
    y = np.linspace(-Ly/2., Ly/2., N)
    X, Y = np.meshgrid(x, y)
    grid = np.array([X.ravel(), Y.ravel(), np.zeros(Y.ravel().shape[0])]).T

    dhist = []; dhist_intrinsic = []
    if plotmols:
        eps = 1.0 #Size to plot either side of domain
        fig_mols = plt.figure()
        ax1 = fig_mols.add_subplot(121)
        ax2 = fig_mols.add_subplot(122, projection='3d')
    #Plot for density
    fig, ax = plt.subplots(1,1); plt.ion()
    plt.show()
    
    n_atoms, Nframes = xyz_num_frames(TRAJ)
    for t in range(100,Nframes):
        print("Reading frame ", t, " of ", Nframes)
        el, xp, yp, zp = read_frame(TRAJ, t) 

        #centre z
        points = np.array([xp-Lx/2., yp-Ly/2., zp-Lz/2.]).T

        #Get liquid and vapour
        labels, liquid = liquid_vapour_dbscan(points[:, :3], eps=1.5, min_samples=8, Lx=Lx, Ly=Ly)
        liquid_pts = points[liquid]
        vapour_pts = points[~liquid]

        #Take just liquid points and fit surface
        q_vectors, modes_shape, Qxy, Q = _compute_q_vectors(box, normal, alpha)
        pivot = np.sort(get_initial_pivots(liquid_pts, box, normal))

        #Keep looping and adding more molecules
        tau_ = tau
        for i in range(Ntries):
            p = liquid_pts[pivot]
            modes = get_surface_modes(p, Qxy, modes_shape)
            s = surface_from_modes(p, q_vectors, modes)
            d = p[:, normal] - s
            new_pivot = np.sort(update_pivots(liquid_pts, q_vectors, modes, pivot, 
                                              normal, alpha, tau_))
            print("No. pivots = ", len(new_pivot), "Error=", np.sqrt(np.sum(d * d) / len(d)), 
                  "tau=", tau_, "Taget pivots=", int(ns * Area))

            if plotmols:
                #if (i == 0 or new_pivot.shape[0] >= int(ns * Area)):
                if (len(new_pivot) != len(pivot)):
                    ax1.cla(); ax2.cla()
                    surf = surface_from_modes(grid, q_vectors, modes, derivative=None).reshape(N, N)
                    meansurfloc = np.mean(surf)
                    rect = patches.Rectangle((-Ly/2, meansurfloc-eps), Ly, eps, fill=False)

                    #Plot surface as it is fitted
                    ax1.scatter(liquid_pts[:,1], liquid_pts[:,2], c="b", s=1.0, label="liquid cluster")
                    ax1.scatter(vapour_pts[:,1], vapour_pts[:,2], c="g", s=1.0, label="vapour")
                    ax1.scatter(p[:,1], p[:,2], c="r", s=3.0, label="edge")
                    ax1.add_patch(rect)
                    ax1.legend()
                    #plt.axis("equal")

                    ax2.scatter(p[:,0], p[:,1], p[:,2], '.', c="r", s=30)
                    ax2.scatter(liquid_pts[:,0], liquid_pts[:,1], liquid_pts[:,2], '.', c="b", s=10.0)
                    ax2.scatter(vapour_pts[:,0], vapour_pts[:,1], vapour_pts[:,2], '.', c="g", s=10.0)
                    ax2.plot_wireframe(X, Y, surf, lw=0.1, color = 'k')
                    #Take small region around domain
                    ax2.set_zlim(meansurfloc-eps, meansurfloc+eps)
                    plt.pause(0.1)

            #Exit if not changing
            if new_pivot.shape[0] >= int(ns * Area):
                break
            elif new_pivot.shape[0] == pivot.shape[0]:
                tau_ += 0.01
            else:
                pivot = new_pivot

        #Shift each particle by surface position
        sp = surface_from_modes(points[:,0:2], q_vectors, modes, derivative=None)
        meansurf = np.mean(sp); d_eps = 5;
        density, edges = np.histogram(points[:,2], bins=nbins, range=(meansurf-d_eps, meansurf+d_eps))
        density_intrinsic, edges_intrinsic = np.histogram(points[:,2]-sp, bins=nbins, range=(-d_eps, d_eps))

        dhist.append(density)
        dhist_intrinsic.append(density_intrinsic)
        ax.cla()
        ax.plot(edges_intrinsic[:-1], np.mean(dhist,0))
        ax.plot(edges[:-1]-meansurf, np.mean(dhist_intrinsic,0))
        ax.set_title("Averaged record = "+str(len(dhist)))
        plt.pause(0.1)


