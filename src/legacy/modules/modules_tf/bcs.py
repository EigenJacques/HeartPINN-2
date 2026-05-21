#%%
# Imports

from numpy import array, pi, sin, cos, where, ones, tile, zeros
from numpy.linalg import norm

from tensorflow import cast, transpose

from pyDOE import lhs
#done
#%%
# Class for the boundary conditions of the PINN
class Collocation:
    def __init__(self, dims):
        """ Initialize the boundary conditions . 
    
        Parameters
        ----------
        dims : nested list
            List of the dimensions of the domain.

        q_top, q_bottom, q_cylinder : float
            geometric "size" of the different boundaries.

        q_wall, q_inl, q_outl, q_total : dictionary
            dictionaries which group different boundary conditions' geometric "sizes".

        d: dictionary
            dictionary with the dimensions of the domain.
    
        """ 

        self.dims = dims

        # Calculating geometric ratios
        q_top       = (dims[0][1] - dims[0][0])*1
        q_bottom    = (dims[0][1] - dims[0][0])*1
        if dims[2][1] == dims[2][0]:
            q_cyl   = pi*dims[2][0]
        else:
            q_cyl   = pi*(dims[2][1] + dims[2][0])*(((dims[2][1]*1)/(dims[2][1] - dims[2][0]))**2 + (dims[2][1] - dims[2][0])**2)**0.5
        q_inl       = (dims[1][1] - dims[1][0])*1
        q_oul       = (dims[1][1] - dims[1][0])*1
        
        q_wall =  { "q_top":q_top, "q_bottom":q_bottom, "q_cyl":q_cyl}
        q_inl =  { "q_inl":q_inl}
        q_outl =  {"q_oul":q_oul}
        
        for q in [q_wall, q_inl, q_outl]:
            q["total"] = 0.0
            for entry in q:
                q["total"] = q["total"] + q[entry]
            q["total"] = q["total"]/2

        # Calculated parameters
        self.q_wall = q_wall
        self.q_inl = q_inl
        self.q_outl = q_outl

        # Parametric functions for the wall boundary condition collocation points.
        def f_wall_top(p1, p2): 
            """ Return a collocation point on the top wall.
            
            Arguments
            ----------
            p1, p2: float
                Random parameter in [0,1]
            
            Returns
            -------
            X: array
                Coordinate of the collocation point.

            N: array
                Normal vector of the collocation point.
            """ 

            x = dims[0][0] + p1*(dims[0][1] - dims[0][0])
            y = dims[1][1]*ones(len(p1))
            z = p2; d = dims[2][0] + z*(dims[2][1] - dims[2][0])

            X = array([x, y, d])

            N = tile(array([[0, -1.0, 0]]).T, len(p1))

            return X, N

        def f_wall_bottom(p1, p2): 
            """ Return a collocation point on the bottom wall.
            
            Arguments
            ----------
            p1, p2: float
                Random parameter in [0,1]
            
            Returns
            -------
            X: array
                Coordinate of the collocation point.

            N: array
                Normal vector of the collocation point.
            """

            x = dims[0][0] + p1*(dims[0][1] - dims[0][0])
            y = dims[1][0]*ones(len(p1))
            z = p2; d = dims[2][0] + z*(dims[2][1] - dims[2][0])

            X = array([x, y, d])

            N = tile(array([[0, 1.0, 0]]).T, len(p1))

            return X, N

        def f_wall_cylinder(p1, p2): 
            """ Return a collocation point on the cylinder.
            
            Arguments
            ----------
            p1, p2: float
                Random parameter in [0,1]
            
            Returns
            -------
            X: array
                Coordinate of the collocation point.

            N: array
                Normal vector of the collocation point.
            """

            z = p2; d = dims[2][0] + z*(dims[2][1] - dims[2][0])
            x = (dims[0][1] - dims[0][0])/2 + dims[0][0] + d*cos(2*pi*p1)
            y = (dims[1][1] - dims[1][0])/2 + dims[1][0] + d*sin(2*pi*p1)
            
            X = array([x, y, d])

            N = array([x-(dims[0][1] - dims[0][0])/2, y-(dims[1][1] - dims[1][0])/2, zeros(x.shape[0])])
            N = N/norm(N, axis=0)
            return X, N

        # Parametric functions for the inlet boundary condition collocation points.
        def f_inlet(p1, p2): 
            """ Return a collocation point on the inlet.
            
            Arguments
            ----------
            p1, p2: float
                Random parameter in [0,1]
            
            Returns
            -------
            X: array
                Coordinate of the collocation point.

            N: array
                Normal vector of the collocation point.
            """

            x = self.dims[0][0]*ones(len(p1))
            y = self.dims[1][0] + p1*(dims[1][1] - dims[1][0])
            z = p2; d = dims[2][0] + z*(dims[2][1] - dims[2][0])

            X = array([x, y, d])
            
            return X

        # Parametric functions for the outlet boundary condition collocation points.    
        def f_outlet(p1, p2): 
            """ Return a collocation point on the outlet.
            
            Arguments
            ----------
            p1, p2: float
                Random parameter in [0,1]
            
            Returns
            -------
            X: array
                Coordinate of the collocation point.

            N: array
                Normal vector of the collocation point.
            """

            x = self.dims[0][1]*ones(len(p1))
            y = self.dims[1][0] + p1*(dims[1][1] - dims[1][0])
            z = p2; d = dims[2][0] + z*(dims[2][1] - dims[2][0])
            
            X = array([x, y, d])
            
            return X

        self.f_wall_top = f_wall_top
        self.f_wall_bottom = f_wall_bottom
        self.f_wall_cylinder = f_wall_cylinder
        self.f_inlet = f_inlet
        self.f_outlet = f_outlet

    def wall(self, n_coll):
        """ Return a collocation point on one of the wall boundaries.

        Arguments
        ----------
        n_coll: int
            number of collocation points to generate

        Parameters
        ----------
        p1: float
            Random parameter in [0,1]. Parametric coordinate along wall length.

        p2: float
            Random parameter in [0,1]. Parametric coordinate along increasing cylinder diameter.

        wheel: list(float)
            Probability wheel to divide up p1 for each wall boundary.
        
        Returns
        -------
        coll_p: array
            Coordinates of the collocation points.

        coll_n: array
            Normal vectors of the collocation points.
        """

        # Probability wheel.
        wheel = [0]
        for i, q in enumerate(self.q_wall):
            wheel.append(wheel[i] + self.q_wall[q]/self.q_wall["total"])
            
            # Break before last entry.
            if i == len(self.q_wall)-2:
                break
        wheel = array(wheel)

        # Random collocation point on a wall.
        p = lhs(2, n_coll)
        p1 = p[:, 0]
        p2 = p[:, 1]
        func = [self.f_wall_top, self.f_wall_bottom, self.f_wall_cylinder]

        coll_P, coll_V = zeros((3,n_coll)), zeros((3,n_coll))
        j = 0
        for i, bc in enumerate(func):    
            coll_ind = where((p1 > wheel[i])*(p1 < wheel[i+1]))[0]
            coll_p, coll_v = bc( (p1[coll_ind] - wheel[i])/(wheel[i+1] - wheel[i]), p2[coll_ind] )
            coll_P[:, j:j+coll_ind.shape[0]] = coll_p
            coll_V[:, j:j+coll_ind.shape[0]] = coll_v
            j += coll_ind.shape[0]

        return coll_P, coll_V

    def inlet(self, n_coll):
        ''' Return collocation point on the inlet.'''
            
        # Probability wheel.
        wheel = [0.0]
        for i, q in enumerate(self.q_inl):
            wheel.append(wheel[i] + self.q_inl[q]/self.q_inl["total"])
            
            # Break before last entry.
            if i == len(self.q_inl)-2:
                break
        wheel = array(wheel)

        # Random collocation point on the inlet.
        p = lhs(2, n_coll)
        p1 = p[:, 0]
        p2 = p[:, 1]
        func = [self.f_inlet]

        for i, bc in enumerate(func):    
            coll_ind = where((p1 > wheel[i])*(p1 < wheel[i+1]))[0]
            coll_p = bc( p1[coll_ind], p2[coll_ind] )
        
        return coll_p

    def outlet(self, n_coll):
        ''' return collocation point on the outlet. Collocation points are solol-randomly distributed. '''
            
        # Probability wheel.
        wheel = [0]
        for i, q in enumerate(self.q_outl):
            wheel.append(wheel[i] + self.q_outl[q]/self.q_outl["total"])
            
            # Break before last entry.
            if i == len(self.q_outl)-2:
                break
        wheel = array(wheel)
        
        # Random collocation point on the outlet.
        p = lhs(2, n_coll)
        p1 = p[:, 0]
        p2 = p[:, 1]
        func = [self.f_outlet]

        for i, bc in enumerate(func):    
            coll_ind = where((p1 > wheel[i])*(p1 < wheel[i+1]))[0]
            coll_p = bc( p1[coll_ind], p2[coll_ind] )
        return coll_p

    def interior(self, n_coll):
        """ Return a collocation point in the interior. Collocation points are sobol-randomly distributed. 

        Arguments
        ----------
    
        Parameters
        ----------
        p1, p2, p3: float
            random variables between 0 and 1. Parametric coordinates corresponding to x, y and d

        x, y: float
            Coordinates in the domain

        d: float
            Diameter of the cylinder. Parametric coordinate.
    
        Returns
        -------
        coll_p : array_like
            coordinate of collocation point. 
        
        """ 

        def fluid(p1, p2, p3):
            """ Return a collocation point in the fluid as a function of p1, p2 and p3 between 0 and 1. """

            x = self.dims[0][0] + p1*(self.dims[0][1] - self.dims[0][0])
            y = self.dims[1][0] + p2*(self.dims[1][1] - self.dims[1][0])
            z = p3; d = self.dims[2][0] + z*(self.dims[2][1] - self.dims[2][0])
            X = array([x, y, d])
            return X

        # Collocation points in the interior
        p = lhs(3, n_coll*2)
        p1 = p[:, 0]
        p2 = p[:, 1]
        p3 = p[:, 2]
        X = fluid(p1, p2, p3)

        d = self.dims[2][0] + X[2,:]*(self.dims[2][1] - self.dims[2][0])
        coll_p = where((X[0,:]-((self.dims[0][1] - self.dims[0][0])/2 + self.dims[0][0]))**2 + (X[1,:]-((self.dims[1][1] - self.dims[1][0])/2 + self.dims[1][0]))**2 > d**2)[0]

        return X[:,coll_p][:,0:n_coll]

def gen_geometry(n_internal, n_wall, n_outlet, n_inlet, d, l_wall, h_wall, precision="float32"):
    geometry = [[0,l_wall],[0,h_wall], [d[0], d[1]]]
    coll = Collocation(dims = geometry)
    X_internal, X_wall, X_inlet, X_outlet, V_wall = list(), list(), list(), list(), list()

    X_internal.append(coll.interior(n_internal))
    X_internal = array(X_internal)[0,:,:]

    X_inlet.append(coll.inlet(n_inlet))
    X_inlet = array(X_inlet)[0,:,:]

    X_outlet.append(coll.outlet(n_outlet))
    X_outlet = array(X_outlet)[0,:,:]

    X_wall, V_wall = coll.wall(n_wall)
    X_wall = array(X_wall)
    V_wall = array(V_wall)

    return transpose(cast(X_internal, dtype=precision)), transpose(cast(X_inlet, dtype=precision)), transpose(cast(X_outlet, dtype=precision)), transpose(cast(X_wall, dtype=precision)), transpose(cast(V_wall, dtype=precision))

#================================================================================================================
# Usage for parametric cylinder
#================================================================================================================
X_internal, X_inlet, X_outlet, X_wall, V_wall = gen_geometry(n_internal=100, n_wall=500, n_outlet=100, n_inlet=100, d=[0.1, 0.2], l_wall=1, h_wall=1)
# # %%
# %matplotlib widget
# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(*X_inlet[0,:,:])
# ax.scatter(*X_outlet[0,:,:])
# ax.scatter(*X_wall[:,:])
# ax.scatter(*X_internal[0,:,:])

# %%
#================================================================================================================
# Usage for regular 2D cylinder
#================================================================================================================
# from matplotlib import pyplot as plt
# X_internal, X_inlet, X_outlet, X_wall, V_wall = gen_geometry(n_internal=100, n_wall=500, n_outlet=100, n_inlet=100, d=[0.1, 0.1], l_wall=1, h_wall=1)

# plt.scatter(*X_wall)
# plt.scatter(*X_inlet)
# plt.scatter(*X_outlet)
# plt.scatter(*X_internal)
# # %%
