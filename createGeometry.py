import numpy as np
import matplotlib.pyplot as plt

# sources
sx_init = 1000
sx_end  = 10000
Nsource = 5
sx = np.linspace(sx_init,sx_end,Nsource,endpoint=False)
sz =  50 * np.ones(len(sx))
sIdx = np.arange(Nsource)

# receivers
rx_init = 0
rx_end = 5000
Nrec = 100
rx = np.linspace(rx_init,rx_end,Nrec,endpoint=False)
rz = 60 * np.ones(len(rx))
rIdx = np.arange(Nrec)


plt.figure()
plt.plot(sx,sz,"r*")
plt.plot(rx,rz,'bv')
plt.xlim(0,10000)
plt.ylim(10000,0)
plt.show(block=False)

receiverTable = np.vstack(np.transpose([rIdx, rx, rz]))
outname="receivers.txt"
np.savetxt(outname,receiverTable,delimiter=",",fmt="%1.2f",header=" index, coord x, coord z")

sourceTable = np.vstack(np.transpose([sIdx, sx, sz]))
outname="sources.txt"
np.savetxt(outname,sourceTable,delimiter=",",fmt="%1.2f",header=" index, coord x, coord z")
