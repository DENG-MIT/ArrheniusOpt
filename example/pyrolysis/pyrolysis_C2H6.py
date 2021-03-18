import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Lhs

import cantera as ct

np.random.seed(1)

gas = ct.Solution('../../mechanism/USCPyrolysis.yaml')
gas.TPX = 1452, 3.32*ct.one_atm, 'C2H6:0.02,Ar:0.98'

Y_fuel_0 = gas.Y[gas.species_index('C2H6')]
print(Y_fuel_0)

r = ct.IdealGasConstPressureReactor(gas, energy='on')
sim = ct.ReactorNet([r])
time = 0.0
states = ct.SolutionArray(gas, extra=['t'])

print('%10s %10s %10s %14s' % ('t [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
for n in range(10000):
    states.append(r.thermo.state, t=sim.time)
    sim.step()

    if r.thermo.Y[gas.species_index('C2H6')] < Y_fuel_0 * 0.05 or sim.time>1.51e-3:
        break

nodedata = np.vstack((states.t, states.T, states.X.T)).T
np.savetxt('pyrolysis_C2H6.dat', nodedata)

plt.clf()
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T, 'o')
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2, 2, 2)
plt.plot(states.t, states.Y[:, gas.species_index('C2H6')])
plt.xlabel('Time (ms)')
plt.ylabel('C2H6 Mole Fraction')
plt.subplot(2, 2, 3)
plt.plot(states.t, states.Y[:, gas.species_index('C2H4')])
plt.xlabel('Time (ms)')
plt.ylabel('C2H4 Mole Fraction')
plt.subplot(2, 2, 4)
plt.plot(states.t, states.Y[:, gas.species_index('C2H2')])
plt.plot(states.t, states.Y[:, gas.species_index('CH4')])
plt.xlabel('Time (ms)')
plt.ylabel('C2H2/CH4 Mole Fraction')
plt.tight_layout()
plt.show()
