"""
Test the use of Brian to create LSM reservoir and train to reproduce sin wave
"""

from brian import *
from ModelSpace import Cuboid3D 
import math
import sys

#### Liquid ####

# Neuron parameters
# http://www.neurdon.com/2011/01/19/neural-modeling-with-python-part-1/
tau_m = 20 * ms	# membrane time constant (R_m * C_m)
Vt = 10 * mV	# spike threshold
Vr = 0 * mV		# spike reset
El = 1 * mV		# leakage (I * R_m)

# Neuron equations (LIF)
eqs_l = Equations('dV/dt = ( -V - El) / tau_m : volt')

# Liquid dimensions
x = 5	# X dimension
y = 5	# Y dimension
z = 5	# Z dimension
N = x * y * z	# Number of Neurons
mspace = Cuboid3D(x, y, z)

# Liquid Neurons
L = NeuronGroup(N, model=eqs_l, threshold=Vt, reset=Vr)



# Need to assign random initial voltages




#### Input ####

# Input Neurons
I = NeuronGroup(2, model='V : volt')


# Neuron parameters (Constant sinewave as input)
A = 5 * mV 			# amplitude
f = 20 * hertz		# frequency
w = 2 * math.pi * f

# Sine wave generator
@network_operation(clock=defaultclock)
def updateInput():
	"""Sine wave"""
	t = defaultclock.t # (ms)
	for i in range(0, len(I.V)):
		phi = (float(i) / 7) * 2 * math.pi # phase
		I.V[i] = A * sin(w*t + phi) # (mV)

#### Monitors ####

# Check sinwave on input
M = StateMonitor(I, 'V', record=True)

# run simulation (default timestep is 0.1 ms)
run (200 * ms)

plot(M.times / ms, M[0] / mV)
plot(M.times / ms, M[1] / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Input Neuron Potential')
show()

