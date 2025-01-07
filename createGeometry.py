import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sources
sx_init = 100
sx_end  = 4900
Nsource = 4800
sx = np.linspace(sx_init,sx_end,Nsource,endpoint=False)
sz = 50 * np.ones(len(sx))
sIdx = np.arange(Nsource)

# receivers
rx_init = 0
rx_end = 5000
Nrec = 5001
rx = np.linspace(rx_init,rx_end,Nrec,endpoint=False)
rz = np.ones(len(rx))
rIdx = np.arange(Nrec)


plt.figure()
plt.plot(sx,sz,"r*")
plt.plot(rx,rz,'bv')
plt.xlim(0,rx_end)
plt.ylim(rx_end,0)
plt.show()

receiver_df = pd.DataFrame({'index': rIdx,'coordx': rx,'coordz': rz})

source_df = pd.DataFrame({'index': sIdx,'coordx': sx,'coordz': sz})

receiver_df.to_csv("d:/GitHub/Geofisica/receivers.csv", index=False)
source_df.to_csv("d:/GitHub/Geofisica/sources.csv", index=False)
