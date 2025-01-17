import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Função para calcular a wavelet de Ricker
def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

# Lendo os arquivos de fontes e receptores
receiverTable = pd.read_csv('d:/GitHub/Geofisica/receivers.csv')
sourceTable = pd.read_csv('d:/GitHub/Geofisica/sources.csv')

rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

# Parâmetros do modelo
L = 10000         # Largura do modelo (m)
H = 3000          # Altura do modelo (m)
T = 2             # Tempo total de simulação (s)
dt = 0.001        # Passo de tempo (s)
dx = dz = 5       # Passo espacial (m)
f0 = 60           # Frequência central da wavelet de Ricker (Hz)

x = np.arange(0, L + dx, dx)  # Coordenadas horizontais (m)
z = np.arange(0, H + dz, dz)  # Coordenadas verticais (m)
t = np.arange(0, T + dt, dt)  # Vetor de tempo (s)

nx = len(x)  # Número de células horizontais
nz = len(z)  # Número de células verticais
nt = len(t)  # Número de passos de tempo

# Modelo de velocidades (três camadas)
v1 = 1500  # Velocidade da camada 1 (m/s)
v2 = 2000  # Velocidade da camada 2 (m/s)
v3 = 3000  # Velocidade da camada 3 (m/s)
v_gr = 780
# Modelo de velocidades em função da profundidade
vp = np.zeros((nz, nx))
vp[0:int(nz / 4), :] = v1
vp[int(nz / 4):int(nz / 2), :] = v2
vp[int(nz / 2):, :] = v3

# Wavelet de Ricker
wavelet = ricker(f0, t)

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
        t_ref = np.sqrt((2 * H / v1) ** 2 + (dist / v1) ** 2)
        t_hw = dist / v2 + (2 * H * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)
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
for i in range(len(shot_x)):
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


plt.ylim(np.max(t_direct_list), 0)
plt.xlabel('Distância')
plt.ylabel('Tempo')
plt.grid(True)
plt.show()

sism = np.zeros((nt, len(rec_x),len(shot_x)))

for s, shot_x_val in enumerate(shot_x):
    for r, rec_x_val in enumerate(rec_x):
        dx = np.abs(rec_x_val - shot_x_val)
        dz = np.abs(rec_z[r] - shot_z[s])
        dist = np.sqrt(dx**2 + dz**2)

        k = int((dist / v1) / dt)  
        y = int((np.sqrt((2 * H / v1) ** 2 + (dist / v1) ** 2)) / dt)  
        z = int((dist / v2 + (2 * H * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)) / dt)  
        u = int((dist / v_gr) / dt)  

        if k < nt:
            sism[k, r, s] = 1
        if y < nt:
            sism[y, r, s] = 1
        if z < nt:
            sism[z, r, s] = 1
        if u < nt:
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

# dist = np.zeros((len(shot_x), len(rec_x)))
# for s in shotindex:
#     for r in recindex:
#         dx = np.abs(rec_x[r] - shot_x[s])
#         dz = np.abs(rec_z[r] - shot_z[s])
#         dist[s, r] = np.sqrt(dx**2 + dz**2)


