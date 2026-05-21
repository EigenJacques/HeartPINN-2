#%%
#=============================================================
# Import modules
#=============================================================

import sys
import os

import tensorflow as tf

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Import local modules
sys.path.insert(0, './modules/modules_tf')
from save_and_recover3 import recover_model3, save_model3
from heartvalve_half import AoValveHalf
from all_all import import_fvm_solution, Network, Pinn

#%%
#=============================================================
# Read logging data
#=============================================================

runname         = "all_rect"
runID           = "3474"
evaluationpath  = f"../output/logging/{runname}{runID}/"
h, w = 2,3

class FiltDict():
    def __init__(self, wanted_keys):
        self.wanted_keys = wanted_keys

    def dictfilt(self, pair):
        """ To filter the entries of cons out of the entries of properties. """

        key, value  = pair
        if key in self.wanted_keys:
            return True  # keep pair in the filtered dictionary
        else:
            return False  # filter pair out of the dictionary


def parse(path):
    """To parse the log file, to extract caseinfo, properties and trainhist."""

    # Parse text file
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    state       = 0
    properties  = {}
    trainhist   = []
    caseinfo    = []
    for line in lines:
        if line  == "==":
            state += 1
        if state == 1:
            if line[0] == "@":
                words = line.split(" ")
                words = [word[1::] for word in words if word[0] == "#"]
                for word in words:
                    word = word.split("$")
                    try:
                        properties[word[1]] = float(word[0])  
                    except:
                        properties[word[1]] = bool(word[0]=="True")
            else:
                caseinfo.append(line)
        if state == 2:
            trainhist.append(line.split(","))
    trainhist = pd.DataFrame(np.array(trainhist[2::]), columns=trainhist[1])

    return caseinfo, properties, trainhist

# Extracting properties and trainhist from the log file
caseinfo, properties, trainhist = parse(evaluationpath+f"{runname}{runID}_log.csv")

# Extracting from properties 
filt = trainhist["runID"].astype(int) == 1
maxepoch = int(trainhist[filt]["iter"].astype(float).max())
cons = dict(filter(FiltDict(wanted_keys = ['mu', 'rho', 'U_inf', 'L', 'alpha']).dictfilt, properties.items()))
methods = dict(filter(FiltDict(wanted_keys = ['hard', 'sipinn', 'gpinn', 'gcn', 'fourier', 'lossweight', 'sp','turb']).dictfilt, properties.items()))
active_bcs = dict(filter(FiltDict(wanted_keys = ['inlet', 'outlet', 'wall', 'symmetry']).dictfilt, properties.items()))
prec = tf.float32
angle = int(properties["angle"])

# Loading the PINN
network = Network(hiddenlayers=int(properties["hiddenlayers"]), nodes=int(properties["nodes_n"]),input_n=2, output_n=int(properties["n_output"]), fourier_sigma=properties["fourier_sigma"], prec=tf.float32, methods=methods)
recover_model3(evaluationpath+f"netparams{maxepoch}_1.json", network, prec)
pinn = Pinn(network, cons, methods, active_bcs)

# Collocation data paths
collocationpaths = [f"../data/valve{angle}.013.pkl",
                    f"../data/valve{angle}.014.pkl",
                    f"../data/valve{angle}.013.pkl",
                    f"../data/valve{angle}.012.pkl",]
wallpath         =  f"../data/wall_float{angle}.0.pkl"
wallpath = f"../data/wall_float{angle}.0.pkl"

# Instantiate collocation data object
valve = AoValveHalf(scaling_l=cons["L"], prec=prec)

#%%
#=============================================================
# Print imported values for verification
#=============================================================
print("#=============================================================")
print(f"hard constraints: \t\t\t {methods['hard']}")
print(f"sipinn connections: \t\t\t {methods['sipinn']}")
print(f"gpinn loss function: \t\t\t {methods['gpinn']}")
print(f"gcn layers: \t\t\t\t {methods['gcn']}")
print(f"fourier encoding layer: \t\t {methods['fourier']}")
print(f"Inverse Dirichlet loss weighting: \t {methods['lossweight']}")
print(f"Streamfunction-Pressure formulation: \t {methods['sp']}")
print(f"Mixing length turbulence model: \t {methods['turb']}")
#%%
print("#=============================================================")
for l in caseinfo:
    print(l)

#%%
print("#=============================================================")
print(f"Characteristic dimensions:")
print(f"\t\t\t\t\t L={cons['L']}")
print(f"\t\t\t\t\t U_inf={cons['U_inf']}")
print(f"\t\t\t\t\t alpha={cons['alpha']}")
print(f"Fluid properties: ")
print(f" \t\t\t\t\t density: {properties['rho']}")
print(f"\t \t\t\t\t viscosity: {properties['mu']}")
print(f"maxepoch: \t\t\t\t {maxepoch}")
print(f"valve angle: \t\t\t\t {angle}")
print(f"hidden layer count = \t\t\t {network.hiddenlayers}")
print(f"node count per layer = \t\t\t {network.nodes}")
print(f"Inlet fatness: \t\t\t\t {properties['fatinlet']}")
print(f"GNN connectivity: \t\t\t {int(properties['graph_connect'])}")
print(f"Fourier sigma: \t\t\t\t {properties['fourier_sigma']}")
#%%
print("#=============================================================")
display(trainhist)
print("#=============================================================")
print("final training loss: ", trainhist["loss_val"].astype(float).iloc[-1])
print("#=============================================================")

# %%
#=============================================================
# Plot training losses
#=============================================================
def gaus(X, Y):

    sigma_prior = 0.5
    
    xmax = np.max(X)
    X = X/xmax*10
    Y = np.log(Y)

    xfit = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)

    kernel = 1**2 * RBF(length_scale=1.0)
    gaussian_process = GaussianProcessRegressor(kernel = kernel, alpha=sigma_prior**2)
    # gaussian_process = GaussianProcessRegressor(alpha=1**2)
    gaussian_process.fit(X, Y)

    yfit, std  = gaussian_process.predict(xfit, return_std=True)

    dyfit = 1.96 * np.sqrt(std)

    xfit = xfit * xmax / 10
    yfit = yfit

    return xfit, yfit, dyfit

limhigh1, limlow1 = 1e5, 1e-2
limhigh2, limlow2 = 1e2, 1e-6

trainhist_time = np.array(trainhist['time'].astype(float))
for i in range(max(trainhist["runID"].astype(int))):
    filt = trainhist["runID"].astype(int) == i+1
    timeoffset = float(trainhist["time"][filt].iloc[0])
    if i == 0:
        print(f"Compile time:{timeoffset}")
    else:
        print(timeoffset)

    trainhist_time[filt] = trainhist['time'][filt].astype(float) - timeoffset

fig1 = plt.figure(figsize=(12,6))
ax11 = fig1.add_subplot(121)
ax12 = fig1.add_subplot(122)

fig2 = plt.figure(figsize=(12,6))
ax21 = fig2.add_subplot(121)
ax22 = fig2.add_subplot(122)

#=============================================================
# validation - iter
X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_test_u'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax11.scatter(X[:,0], Y, s=5)
ax11.plot(xfit, np.exp(yfit), color='black')
ax11.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_u",)

X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_test_v'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax11.scatter(X[:,0], Y, s=5)
ax11.plot(xfit, np.exp(yfit), color='black')
ax11.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_v",)

X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_test_p'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax11.scatter(X[:,0], Y, s=5)
ax11.plot(xfit, np.exp(yfit), color='black')
ax11.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_p",)

ax11.set_yscale("log")
ax11.legend()
ax11.set_ylim((limlow1, limhigh1))

#=============================================================
# loss - iter
X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_fluid'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax12.scatter(X[:,0], Y, s=5)
ax12.plot(xfit, np.exp(yfit), color='black')
ax12.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_fluid",)

X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_inlet'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax12.scatter(X[:,0], Y, s=5)
ax12.plot(xfit, np.exp(yfit), color='black')
ax12.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_in",)

X, Y    = [np.array(trainhist['iter'].astype(float)).reshape(-1, 1),
          np.array(trainhist['l_outlet'].astype(float))]
Y       = Y+1E3
xfit, yfit, dyfit = gaus(X, Y)
yfit    = yfit-1E3
Y       = Y-1E3
ax12.scatter(X[:,0], Y, s=5)
ax12.plot(xfit, np.exp(yfit), color='black')
ax12.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_out",)

X, Y = np.array(trainhist['iter'].astype(float)).reshape(-1, 1), np.array(trainhist['l_wall'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax12.scatter(X[:,0], Y, s=5)
ax12.plot(xfit, np.exp(yfit), color='black')
ax12.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_wall",)

ax12.set_yscale("log")
ax12.legend()
ax12.set_ylim((limlow2, limhigh2))

#=============================================================
# validation - time
X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_test_u'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax21.scatter(X[:,0], Y, s=5)
ax21.plot(xfit, np.exp(yfit), color='black')
ax21.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_u",)

X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_test_v'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax21.scatter(X[:,0], Y, s=5)
ax21.plot(xfit, np.exp(yfit), color='black')
ax21.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_v",)

X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_test_p'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax21.scatter(X[:,0], Y, s=5)
ax21.plot(xfit, np.exp(yfit), color='black')
ax21.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_test_p",)

ax21.set_yscale("log")
ax21.legend()


#=============================================================
# loss - time
X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_fluid'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax22.scatter(X[:,0], Y, s=5)
ax22.plot(xfit, np.exp(yfit), color='black')
ax22.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_fluid",)

X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_inlet'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax22.scatter(X[:,0], Y, s=5)
ax22.plot(xfit, np.exp(yfit), color='black')
ax22.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_in",)

X, Y    = [trainhist_time.reshape(-1, 1),
          np.array(trainhist['l_outlet'].astype(float))]
Y       = Y+1E3
xfit, yfit, dyfit = gaus(X, Y)
yfit    = yfit-1E3
Y       = Y-1E3
ax22.scatter(X[:,0], Y, s=5)
ax22.plot(xfit, np.exp(yfit), color='black')
ax22.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_out",)

X, Y = trainhist_time.reshape(-1, 1), np.array(trainhist['l_wall'].astype(float))
xfit, yfit, dyfit = gaus(X, Y)
ax22.scatter(X[:,0], Y, s=5)
ax22.plot(xfit, np.exp(yfit), color='black')
ax22.fill_between(xfit.ravel(),np.exp(yfit - 1.96 * dyfit),np.exp(yfit + 1.96 * dyfit),alpha=0.3,label=r"l_wall",)

ax22.set_yscale("log")
ax22.legend()
ax22.set_ylim((limlow2, limhigh2))


# %%
#=============================================================
# Illustrate solution
#=============================================================
try:
    Xs.keys()
except:
    # Import/generate collocation data
    valve           = AoValveHalf(scaling_l=cons["L"], prec=prec)
    valve.condargs = {"sigma": 10.0, "threshold": 1000, "smoothing":0.001}
    valve.init_rect(h, w)
    Xs              = valve.rect_coll(ncol=[10,5,5,5])
    Xs = list(Xs)

    Xs[0], [F_in, F_out, F_wall, dF_in, dF_out, dF_wall, w_in, w_out, w_wall, dw_in, dw_out, dw_wall], _  = valve.rect_floaters_hard(Xs[0], properties["fatinlet"], [], do_condition=methods["hard"])

    A_inlet, D_inlet, Ah_inlet = valve.mesh(Xs[1], n=2)
    A_outlet, D_outlet, Ah_outlet = valve.mesh(Xs[2], n=2)
    A_wall, D_wall, Ah_wall = valve.mesh(Xs[3], n=2)

    D_fluid = valve.rect_floaters_distances(Xs, kind='wall')
    print(f"Max wall distance: {tf.reduce_max(D_fluid)}")
    cons['dmax'] = tf.reduce_max(D_fluid)

    # Import/generate validation data
    # Validation data import
    X_fvm, Y_fvm = import_fvm_solution(cons, "../data/FVM_laminar.csv", 1000, prec=prec, unitsin="mm")

    X_fvm, [F_in_fvm, F_out_fvm, F_wall_fvm, dF_in_fvm, dF_out_fvm, dF_wall_fvm, w_in_fvm, w_out_fvm, w_wall_fvm, dw_in_fvm, dw_out_fvm, dw_wall_fvm], Y_fvm_l = valve.rect_floaters_hard(X_fvm, properties["fatinlet"], [Y_fvm[:,0:1], Y_fvm[:,1:2], Y_fvm[:,2:3]], do_condition=methods["hard"])

    A_fvm, D_fvm, Ah_fvm = valve.mesh(X_fvm, n=int(properties['graph_connect']))

    # Assemble Xs dictionary
    Xs = {"fluid":{"X":Xs[0], "D":D_fluid,
                    "F_in":F_in, "F_out":F_out, "F_wall":F_wall,
                    "dF_in":dF_in,"dF_out":dF_out, "dF_wall":dF_wall,
                    "w_in":w_in, "w_out":w_out,"w_wall":w_wall, 
                    "dw_in":dw_in, "dw_out":dw_out, "dw_wall":dw_wall},
            "inlet":{"X":Xs[1][:,0:2], "A_hat":Ah_inlet},
            "outlet":{"X":Xs[2][:,0:2], "A_hat":Ah_outlet},
            "wall":{"X":Xs[3][:,0:2], "A_hat":Ah_wall},
            "validation":{"X":X_fvm, "F_in":F_in_fvm, "F_out":F_out_fvm, 
                        "F_wall":F_wall_fvm,"dF_in":dF_in_fvm,
                        "w_in":w_in_fvm, "w_out":w_out_fvm,"w_wall":w_wall_fvm,
                        "dw_in":dw_in_fvm,"dw_out":dw_out_fvm, 
                        "dw_wall":dw_wall_fvm,"U":Y_fvm_l[0], 
                        "V":Y_fvm_l[1], "P":Y_fvm_l[2],"A_hat":Ah_fvm}}
    
X = Xs["fluid"]["X"]
A_hat = valve.mesh(Xs["fluid"]["X"], n=int(properties['graph_connect']))[2]

# # Plot predicted solution
print("#=============================================================")
print("Predicted solution")
with tf.GradientTape(persistent=True) as tape:
    tape.watch(X)
    U, V, P = pinn.formulation(X, A_hat, tape)
dU = tape.gradient(U, X)
dV = tape.gradient(V, X)
dP = tape.gradient(P, X)
del tape 

if methods["hard"]:
    U, V, P = pinn.hard_transform(X, U, V, P, dU, dV, dP, Xs["fluid"])

fig = plt.figure(figsize = (6,8))
ax1 = fig.add_subplot(311)
            
im1     = ax1.scatter(Xs["fluid"]["X"][:,0], Xs["fluid"]["X"][:,1], c=U[:,0], marker = ".", s=60); ax1.set_title("u")
divider = make_axes_locatable(ax1)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation="vertical")

ax2     = fig.add_subplot(312)
im2     = ax2.scatter(Xs["fluid"]["X"][:,0], Xs["fluid"]["X"][:,1], c=V[:,0], marker = ".", s=60); ax2.set_title("v")
divider = make_axes_locatable(ax2)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation="vertical")

ax3     = fig.add_subplot(313)
im3     = ax3.scatter(Xs["fluid"]["X"][:,0], Xs["fluid"]["X"][:,1], c=P[:,0], marker = ".", s=60); ax3.set_title("p")
divider = make_axes_locatable(ax3)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation="vertical")
plt.show()

#Plot fvm/ground truth solution
print("#=============================================================")
print("FVM solution")
X = Xs["validation"]["X"]

fig = plt.figure(figsize = (6,8))
ax1 = fig.add_subplot(311)
            
im1     = ax1.scatter(X[:,0], X[:,1], c=Xs["validation"]["U"], marker = ".", s=60); ax1.set_title("u")
divider = make_axes_locatable(ax1)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation="vertical")

ax2     = fig.add_subplot(312)
im2     = ax2.scatter(X[:,0], X[:,1], c=Xs["validation"]["V"], marker = ".", s=60); ax2.set_title("v")
divider = make_axes_locatable(ax2)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation="vertical")

ax3     = fig.add_subplot(313)
im3     = ax3.scatter(X[:,0], X[:,1], c=Xs["validation"]["P"], marker = ".", s=60); ax3.set_title("p")
divider = make_axes_locatable(ax3)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation="vertical")
plt.show()

# Plot difference
print("#=============================================================")
print("Difference")
with tf.GradientTape(persistent=True) as tape:
    tape.watch(X)
    U, V, P = pinn.formulation(X, A_hat, tape)
dU = tape.gradient(U, X)
dV = tape.gradient(V, X)
dP = tape.gradient(P, X)
del tape

if methods["hard"]:
    U, V, P = pinn.hard_transform(X, U, V, P, dU, dV, dP, Xs["validation"])

fig = plt.figure(figsize = (6,8))
ax1 = fig.add_subplot(311)
            
im1     = ax1.scatter(X[:,0], X[:,1], c=abs(U[:,0]-Xs["validation"]["U"][:,0]), marker = ".", s=60); ax1.set_title("u")
divider = make_axes_locatable(ax1)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation="vertical")

ax2     = fig.add_subplot(312)
im2     = ax2.scatter(X[:,0], X[:,1], c=abs(V[:,0]-Xs["validation"]["V"][:,0]), marker = ".", s=60); ax2.set_title("v")
divider = make_axes_locatable(ax2)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation="vertical")

ax3     = fig.add_subplot(313)
im3     = ax3.scatter(X[:,0], X[:,1], c=abs(P[:,0]-Xs["validation"]["P"][:,0]), marker = ".", s=60); ax3.set_title("p")
divider = make_axes_locatable(ax3)
cax     = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation="vertical")
plt.show()

# %%
#=============================================================
# Training animation
#=============================================================

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, X, datatoplot):

        self.colormap   = matplotlib.cm.viridis
        self.x          = X
        self.datatoplot = datatoplot

        

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        print(len(self.datatoplot[0]))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10, 
                                          init_func=self.setup_plot, blit=True, frames=len(self.datatoplot[0])-1)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
                    
        self.scat = self.ax.scatter(self.x[:,0], self.x[:,1], c=self.datatoplot[0][0][:,0])

        return self.scat,

    def update(self, i):
        """Update the scatter plot."""

        self.scat = self.ax.scatter(self.x[:,0], self.x[:,1], c=self.datatoplot[0][i][:,0])
        self.ax.set_title(f"{i}")

        return self.scat,

try:
    len(Us)
except:
    Us = []
    Vs = []
    Ps = []

    X = Xs["validation"]["X"]

    filt = trainhist["runID"].astype(int) == 1

    for ep in np.array(trainhist["iter"][filt].astype(float), dtype=int):

        network = Network(hiddenlayers=int(properties["hiddenlayers"]), nodes=int(properties["nodes_n"]),input_n=2, output_n=3, fourier_sigma=1, prec=tf.float32, methods=methods)
        recover_model3(evaluationpath+f"netparams{ep}_1.json", network, prec)
        pinn = Pinn(network, cons)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            U, V, P = network.predict(X)
        dU = tape.gradient(U, X)
        dV = tape.gradient(V, X)
        dP = tape.gradient(P, X)
        del tape

        U, V, P = pinn.hard_transform(X, U, V, P, dU, dV, dP, Xs["validation"])

        Us.append(U)
        Vs.append(V)
        Ps.append(P)

# %%

a = AnimatedScatter(X, [Us, Vs, Ps])
plt.show()
animate = a.ani
f = "../output/animation_network.mp4" 
writervideo = animation.FFMpegWriter(fps=10) 
animate.save(f, writer = writervideo)
# %%