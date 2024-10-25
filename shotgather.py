import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import ricker

Nx = 501
dx_rec = 25
Nz = 1001
dz = 1

rec_x = np.arange(0, Nx, dx_rec) 
recindex = np.arange(1, len(rec_x) + 1)
shot_x = [0, (Nx-1) / 2]
shotindex = [1, 2]

index = np.concatenate((recindex, shotindex))
tabela = np.zeros((len(index), 3))
##### NEW

inputFileReceiver = "receivers.txt"
receiverTable = np.loadtxt(inputFileReceiver,delimiter=",",skiprows=1)

inputFileSource = "sources.txt"
sourceTable = np.loadtxt(inputFileSource,delimiter=",",skiprows=1)

raise ValueError



for i in range(len(rec_x)):
    tabela[i, 0] = recindex[i]   
    tabela[i, 1] = rec_x[i]  
    tabela[i, 2] = 0     

for i in range(len(shot_x)):
    tabela[len(recindex) + i, 0] = shotindex[i] 
    tabela[len(recindex) + i, 1] = shot_x[i] 
    tabela[len(recindex) + i, 2] = 0   

df_tabela = pd.DataFrame(tabela, columns=["Index", "x (m)", "z (m)"])
print(df_tabela)

plt.scatter(tabela[:len(rec_x), 1], tabela[:len(rec_x), 2], c='blue', label='Receptores')
plt.scatter(tabela[len(rec_x):, 1], tabela[len(rec_x):, 2], c='red', label='Fontes (Shots)')
plt.xlabel('Distância horizontal (m)')
plt.ylabel('Profundidade (m)')
plt.ylim(0, Nz)
plt.gca().invert_yaxis()
plt.title('Posições de Receptores e Fontes (Shots)')
plt.legend()
plt.grid(True)
plt.show()   


Nt = 1001
dt = 0.001
x = np.linspace(0,dx_rec*(Nx-1),dx_rec)
t = np.linspace(0,dt*(Nt-1),Nt)

t1, t2 = 0.3, 0.6
rho1, rho2, rho3 = 2200, 2400, 2700
vp1, vp2, vp3 = 2500, 3000, 3500
z1, z2, z3 = rho1 * vp1, rho2 * vp2, rho3 * vp3

rho = np.zeros(Nt)
vp  = np.zeros(Nt)

rho[:int(t1/dt)]           = rho1
rho[int(t1/dt):int(t2/dt)] = rho2
rho[int(t2/dt):]           = rho3

vp[:int(t1/dt)]           = vp1
vp[int(t1/dt):int(t2/dt)] = vp2
vp[int(t2/dt):]           = vp3

Z = vp * rho


z2D = np.zeros([Nt, Nx])
for ix in range(Nx):
    z2D[:, ix] = Z


plt.figure()
plt.imshow(z2D, cmap="jet", aspect="auto", extent=[0, dx_rec * (Nx - 1), t[-1], t[0]])
plt.xlabel("Distância lateral (m)")
plt.xlim(0, Nx)
plt.ylabel("Tempo (s)")
plt.title("Impedância")
plt.colorbar()


plt.scatter(tabela[:, 1], tabela[:, 2], c='red', marker='o', label='Receptores e Fontes')
plt.legend()
plt.tight_layout()
plt.show()

# tive que aumentar as velocidades para que aparecessem os tempos de trânsito
h = Nt - 1
v1 = 15000
v2 = 40000
v_gr= 4000
wavelet = ricker(Nt, 4)
x = np.linspace(0,dx_rec*(Nx-1),Nx)

t_ref = np.sqrt((2*h/v1)**2 + (x/v1)**2)
t_direct = x/v1
t_hw = x/v2 + (2*h*np.sqrt(v2**2-v1**2))/(v1*v2)
t_gr= x/v_gr
sism = np.zeros((Nt, Nx))

plt.plot(x, t_direct, label = "direct wave")
plt.plot(x,t_ref, label = "reflection wave")
plt.plot(x,t_hw, label = "head wave")
plt.plot(x, t_gr, label = "ground role")
plt.ylim(t[-1],0)
plt.xlabel('Distância horizontal (m)')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.legend()
plt.show()

sism = np.zeros((Nt,Nx))

for ix in range(Nx):
    k = int(t_direct[ix]/dt)
    y = int(t_ref[ix]/dt)
    z = int(t_hw[ix]/dt)
    u = int(t_gr[ix]/dt)

    sism[k,ix] = 1
    sism[y,ix] = 1
    sism[z,ix] = 1
    if u < Nt:
        sism[u,ix] = 1

for x in range(Nx):
    sism[:, x] = np.convolve(sism[:, x], wavelet, mode='same')

k = np.max(np.abs(sism))

plt.figure(figsize=(10, 6))
plt.imshow(sism, cmap="seismic", aspect="auto", extent=[0, dx_rec*(Nx-1), t[-1], t[0]], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.title("Sismograma")
plt.tight_layout()
plt.show()


dist = np.zeros((len(shot_x), len(rec_x)))

for i, shot in enumerate(shot_x): 
    for j, receptor in enumerate(rec_x): 
        dist[i, j] = np.abs(receptor - shot)


df_dist = pd.DataFrame(dist, columns=[f'Receptor {j+1}' for j in range(len(rec_x))], index=[f'Shot {i+1}' for i in range(len(shot_x))])

print(df_dist)