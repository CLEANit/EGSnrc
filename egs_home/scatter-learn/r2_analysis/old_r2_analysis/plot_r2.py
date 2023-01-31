import matplotlib.pyplot as plt
import numpy as np
results = [[500,0.6064792761345765],
           [1000,0.7314659046091694],
           [5000,0.9203924405920161],
           [10000,0.9554927095259401],
           [50000,0.9785541736805392],
           [100000,0.9794629796186345],
           [500000,0.9860766890849111]]

results = np.array(results)

plt.figure()
plt.title("r2 Values by Number of Particles")
plt.xlabel("Log Number of Particles")
plt.ylabel("r2 Value")
plt.semilogx(results[:,0],results[:,1], 'o', c='b')
plt.semilogx(results[:,0],results[:,1], c='b')
plt.savefig("r2_values_by_ncase.png")
#plt.show()
plt.close()
