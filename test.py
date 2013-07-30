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
tau_m_l = 20 * ms	# membrane time constant (R_m * C_m)
Vt_l = 10 * mV	# spike threshold
Vr_l = 0 * mV		# spike reset
El = 1 * mV		# leakage (I * R_m)

# Neuron equations (LIF)
eqs_l = Equations('dV/dt = ( -V - El) / tau_m_l : volt')

# Liquid dimensions
x = 5	# X dimension
y = 5	# Y dimension
z = 5	# Z dimension
N_l = x * y * z	# Number of Neurons
mspace = Cuboid3D(x, y, z)

# Liquid Neurons
L = NeuronGroup(N_l, model=eqs_l, threshold=Vt_l, reset=Vr_l)


#### Input ####

## NOTE: better off using a neuron subgroup

# Input Neurons
N_i = 2 			# Number of input neurons
tau_m_i = 20 * ms 	# membrane timeconstant
Vt_i = 2.5 * mV		# spike threshold
Vr_i = 0 * mV		# spike reset

# Vi is the ingection voltage from the signal source
eqs_i = Equations("""
	dV/dt = ( -V + Vi) / tau_m_i : volt
	Vi : volt
""")

# Define neuron group
I = NeuronGroup(2, model=eqs_i, threshold=Vt_i, reset=Vr_l)

# Assign random initial voltage
I.V = Vr_i + rand(N_i) * (Vt_i - Vr_i)

# Neuron parameters (Constant sinewave as input)
A = 10 * mV 			# amplitude
f = 20 * hertz			# frequency
w = 2 * math.pi * f

# Sine wave generator
@network_operation(clock=defaultclock)
def updateInput():
	"""Sine wave"""
	t = defaultclock.t # (ms)
	for i in range(0, len(I.V)):
		phi = (float(i) / 7) * 2 * math.pi # phase
		I.Vi[i] = A * sin(w*t + phi) # (mV)

#### Monitors ####

# Check sinwave on input
M_V = StateMonitor(I, 'V', record=True)
M_Vi = StateMonitor(I, 'Vi', record=True)

# run simulation (default timestep is 0.1 ms)
run (200 * ms)

# Plotting
figure()
subplots_adjust(hspace=0.5)
subplot(211)
plot(M_V.times / ms, M_V[0] / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Input Neuron Potential')
subplot(212)
plot(M_Vi.times / ms, M_Vi[0] / mV)
#plot(M.times / ms, M[1] / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Input Neuron Injection Potential')
show()

