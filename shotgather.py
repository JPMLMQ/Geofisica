import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import ricker


receiverTable = pd.read_csv('d:/GitHub/Geofisica/receivers.csv')
sourceTable = pd.read_csv('d:/GitHub/Geofisica/sources.csv')

rec_x = receiverTable['coordx']
rec_z = receiverTable['coordz']
recindex = receiverTable['index']
rec_z = receiverTable['coordz']
shot_x = sourceTable['coordx']
shot_z = sourceTable['coordz']
shotindex = sourceTable['index']
shot_z = sourceTable['coordz']

Nz= 1000

plt.scatter(rec_x, rec_z, color='red', marker='*',label='Receptores')
plt.scatter(shot_x, shot_z, color='blue', marker='v',label='Fontes')
plt.xlabel('Distância horizontal (m)')
plt.ylabel('Profundidade (m)')
plt.ylim(0, Nz)
plt.gca().invert_yaxis()
plt.title('Posições de Receptores e Fontes')
plt.legend()
plt.grid(True)
plt.show()   

rx_init = int(rec_x.iloc[0])
rx_end = int(rec_x.iloc[-1])
x = np.linspace(rx_init,rx_end,len(rec_x),endpoint=False)

Nt = 1001
dt = 0.001
t = np.linspace(0,dt*(Nt-1),Nt)

v1 = 15000
v2 = 40000
v_gr= 4000
wavelet = ricker(Nt, 4)

t_direct_list = []
t_ref_list = []
t_hw_list = []
t_gr_list = []

for s, shot_x_val in enumerate(shot_x):
    t_direct_wave = []
    t_ref_wave = []
    t_hw_wave = []
    t_gr_wave = []
    
    for r, rec_x_val in enumerate(rec_x):
        dx = np.abs(rec_x_val - shot_x_val)
        dz = np.abs(rec_z[r] - shot_z[s])
        dist = np.sqrt(dx**2 + dz**2)
        
        t_direct = dist / v1
        t_ref = np.sqrt((2 * Nz / v1) ** 2 + (dist / v1) ** 2)
        t_hw = dist / v2 + (2 * Nz * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)
        t_gr = dist / v_gr

        t_direct_wave.append(t_direct)
        t_ref_wave.append(t_ref)
        t_hw_wave.append(t_hw)
        t_gr_wave.append(t_gr)
     
    t_direct_list.append(t_direct_wave)
    t_ref_list.append(t_ref_wave)
    t_hw_list.append(t_hw_wave)
    t_gr_list.append(t_gr_wave)


# QC Quality
i = 9
plt.figure()
plt.title(" shot %s"%i)
plt.plot(rec_x, t_direct_list[i], label="Direct wave")
plt.plot(rec_x, t_ref_list[i], label="Reflection wave")
plt.plot(rec_x, t_hw_list[i], label="Head wave")
plt.plot(rec_x, t_gr_list[i], label="Ground roll")

plt.ylim(np.max(t_direct_list), 0)
plt.xlabel('Distância')
plt.ylabel('Tempo')
plt.grid(True)
plt.show()

sism = np.zeros((Nt, len(rec_x),len(shot_x)))

for s, shot_x_val in enumerate(shot_x):
    for r, rec_x_val in enumerate(rec_x):
        dx = np.abs(rec_x_val - shot_x_val)
        dz = np.abs(rec_z[r] - shot_z[s])
        dist = np.sqrt(dx**2 + dz**2)

        k = int((dist / v1) / dt)  
        y = int((np.sqrt((2 * Nz / v1) ** 2 + (dist / v1) ** 2)) / dt)  
        z = int((dist / v2 + (2 * Nz * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)) / dt)  
        u = int((dist / v_gr) / dt)  

        if k < Nt:
            sism[k, r, s] = 1
        if y < Nt:
            sism[y, r, s] = 1
        if z < Nt:
            sism[z, r, s] = 1
        if u < Nt:
            sism[u, r, s] = 1

    for r in range(len(rec_x)):
        sism[:, r, s] = np.convolve(sism[:, r, s], wavelet, mode='same')



# QC Quality
i = 9
k = np.max(np.abs(sism[:,:,i]))   
plt.figure()
plt.title(" shot %s"%i)
plt.imshow(sism[:,:,i], cmap="seismic", aspect="auto", extent=[0, len(rec_x), t[-1], t[0]], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.xlabel('Receptores')
plt.ylabel('Tempo (s)')
plt.tight_layout()
plt.show()

dist = np.zeros((len(shot_x), len(rec_x)))
for s in shotindex:
    for r in recindex:
        dx = np.abs(rec_x[r] - shot_x[s])
        dz = np.abs(rec_z[r] - shot_z[s])
        dist[s, r] = np.sqrt(dx**2 + dz**2)


