"""
Test the use of Brian to create LSM reservoir and train to reproduce sin wave
"""

from brian import *
from ModelSpace import Point3D, Cuboid3D 
import math
import sys

#### Liquid ####

# Neuron parameters
# http://www.neurdon.com/2011/01/19/neural-modeling-with-python-part-1/
tau_m_l = 20 * ms	# membrane time constant (R_m * C_m)
Vt_l = -50 * mV		# spike threshold
Vr_l = -58 * mV		# spike reset
El = -51 * mV		# Rest potential (I * R_m)
psp_l = 0.75 * mV	# Post-synaptic potential

# Neuron equations (LIF)
eqs_l = Equations('dV/dt = (-V + El) / tau_m_l : volt')

# Liquid dimensions
x = 5	# X dimension
y = 5	# Y dimension
z = 5	# Z dimension
N_l = x * y * z	# Number of Neurons
mspace_l = Cuboid3D(x, y, z)

# Liquid Neurons
L = NeuronGroup(N_l, model=eqs_l, threshold=Vt_l, reset=Vr_l)

# Assign random initial voltage
L.V = Vr_l + rand(N_l) * (Vt_l - Vr_l)

# Connect Liquid neurons together
Cl = Connection(L, L)
Cl.connect_random(sparseness=0.1, weight=psp_l)

# Monitors
Ml_s = SpikeMonitor(L)
Ml_V = StateMonitor(L, 'V', record=0)

# Spike generator
N_g = 5 		#Number of spiking neurons
spiketimes = []
#spiketimes = [(0, 10 * ms)]
spiketimes = spiketimes + [(0, x * ms) for x in range(0, 10)]
#spiketimes = spiketimes + [(1, x * ms) for x in range(1, 9)]
#spiketimes = spiketimes + [(2, x * ms) for x in range(2, 8)]
#spiketimes = spiketimes + [(3, x * ms) for x in range(3, 7)]
#spiketimes = spiketimes + [(4, x * ms) for x in range(4, 5)]
G = SpikeGeneratorGroup(N_g, spiketimes)
Cg = Connection(G, L)
Cg.connect_random(sparseness=0.2, weight=psp_l)

#### Input ####

# Input dimensions
x = 1	# X dimension
y = 5	# Y dimension
z = 1	# Z dimension
N_i = x * y * z	# Number of Neurons
offset = Point3D(-2,0,0) # place input neurons next to liquid 
mspace_i = Cuboid3D(x, y, z, offset=offset)

# Neurons Parameters
tau_m_i = 20 * ms 	# membrane timeconstant
Vt_i = 2.5 * mV		# spike threshold
Vr_i = 0 * mV		# spike reset

# Vi is the injection voltage from the signal source
eqs_i = Equations("""
	dV/dt = ( -V + Vi) / tau_m_i : volt
	Vi : volt
""")

# Define neuron group
I = NeuronGroup(N_i, model=eqs_i, threshold=Vt_i, reset=Vr_l)

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
Mi_V = StateMonitor(I, 'V', record=True)
Mi_Vi = StateMonitor(I, 'Vi', record=True)

# run simulation (default timestep is 0.1 ms)
run (100 * ms)

# Plotting
figure()
subplots_adjust(hspace=0.5)
subplot(211)
raster_plot(Ml_s, title="Liquid", newfigure=False)
subplot(212)
plot(Ml_V.times / ms, Ml_V[0] / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Liquid Neuron Potential')
show()

# Plotting
#figure()
#subplots_adjust(hspace=0.5)
#subplot(211)
#plot(M_V.times / ms, Mi_V[0] / mV)
#xlabel('Time (ms)')
#ylabel('Voltage (mV)')
#title('Input Neuron Potential')
#subplot(212)
#plot(M_Vi.times / ms, Mi_Vi[0] / mV)
#xlabel('Time (ms)')
#ylabel('Voltage (mV)')
#title('Input Neuron Injection Potential')
#show()

