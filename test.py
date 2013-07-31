"""
Test the use of Brian to create LSM reservoir and train to reproduce sin wave
"""

from brian import *
from ModelSpace import Point3D, Cuboid3D 
from numpy import *
import math
import sys

### Simulation Parameters

dt = defaultclock.dt # timestep (default is 0.1 ms)
simtime = 200 * ms 
nsteps = simtime / dt

### NOTES ####

# - Input neurons have no internal connections
# - Input neurons have no random initial voltage
# - Input neurons have no refractory period

#### Liquid ####

# Neuron parameters
# http://www.neurdon.com/2011/01/19/neural-modeling-with-python-part-1/
L_tau_m = 20 * ms	# liquid membrane time constant (R_m * C_m)
L_tau_e = 4 * ms	# liquid excitatory synaptic time constant
L_tau_i = 10 * ms	# liquid inhibitory synaptic time constant
L_V_t = -51 * mV	# liquid spike threshold
L_V_r = -57 * mV	# liquid spike reset
L_V_eq = -52 * mV	# liquid equilibrium potential (I * R_m)
L_w_e = 1.5 * mV	# excitatory synaptic weight
L_w_i = 4 * mV		# inhibitory synaptic weight

# Neuron equations (LIF)
L_eqs = Equations("""
	dV/dt = (-V + L_V_eq + ge - gi) / L_tau_m : volt
	dge/dt = -ge/L_tau_e : volt
	dgi/dt = -gi/L_tau_i : volt
""")

# Liquid dimensions
L_x = 5	# X dimension
L_y = 5	# Y dimension
L_z = 5	# Z dimension
L_n = L_x * L_y * L_z	# Number of Neurons
L_n_e = int(math.ceil(0.75 * L_n))	# Number of excitatory neurons
L_n_i = L_n - L_n_e		# Number of inhibitory neurons
L_mspace = Cuboid3D(L_x, L_y, L_z)

# Liquid neuron groups
L = NeuronGroup(L_n, model=L_eqs, threshold=L_V_t, reset=L_V_r, refractory=2 * ms)
L_e = L.subgroup(L_n_e)	# excitatory neurons
L_i = L.subgroup(L_n_i) # inhibitory neurons

# Assign random initial voltage
L.V = L_V_r + rand(L_n) * (L_V_t - L_V_r)

# Connect Liquid neurons together
L_C_e = Connection(L_e, L, 'ge', sparseness=0.6, weight=L_w_e)
L_C_i = Connection(L_i, L, 'gi', sparseness=0.4, weight=L_w_e)

# Monitors
L_M_s = SpikeMonitor(L)
L_M_V = StateMonitor(L, 'V', record=0)
L_M_ge = StateMonitor(L, 'ge', record=0)
L_M_gi = StateMonitor(L, 'gi', record=0)

# Spike generator
#iN_g = 5 		#Number of spiking neurons
#spiketimes = []
#spiketimes = [(0, 10 * ms)]
#spiketimes = spiketimes + [(0, x * ms) for x in range(0, 10)]
#G = SpikeGeneratorGroup(N_g, spiketimes)
#Cg = Connection(G, L)
#Cg.connect_random(sparseness=0.5, weight=0.75 * mV)

##### Input ####

# Neurons Parameters
I_tau_m = 20 * ms	# input membrane time constant
I_V_t = 2.5 * mV	# input spike threshold
I_V_r = -3 * mV		# input spike reset
I_V_eq = 1 * mV		# input rest potential
I_psp = 1.0 * mV	# input neuron post-synaptic potential

# I_V_j is the injection voltage from the signal source
I_eqs = Equations("""
	dV/dt = ( -V + I_V_j + I_V_eq) / I_tau_m : volt
	I_V_j : volt
""")

# Input dimensions
I_x = 1	# X dimension
I_y = 1	# Y dimension
I_z = 1	# Z dimension
I_n = I_x * I_y * I_z		# number of input neurons
I_offset = Point3D(-2,0,0) 	# place input neurons next to liquid 
I_mspace = Cuboid3D(I_x, I_y, I_z, offset=I_offset)

# Define neuron group
I = NeuronGroup(I_n, model=I_eqs, threshold=I_V_t, reset=I_V_r)

# Assign random initial voltage
#I.V = I_V_r + rand(I_n) * (I_V_t - I_V_r)

# Input neuron group connections
# -- currently no interal connections
# May want to create inhibitory and excitatory currents from input to liquid
I_C_L = Connection(I, L, 'V', sparseness=0.6, weight=I_psp)


# Signal parameters (Constant sinewave as input)
A = 10 * mV 			# amplitude
c = 5 * mV				# vertical shift
f = 40 * hertz			# frequency
w = 2 * math.pi * f

# Signal generator
signal_x = [] # x values
signal_y = [] # y values
@network_operation(clock=defaultclock)
def updateInput():
	"""Sine wave"""
	t = defaultclock.t # (ms)
	for i in range(0, len(I.V)):
		#phi = (float(i) / 7) * 2 * math.pi # phase
		phi = 0
		y = A * sin(w*t + phi) + c # (mV)	
		signal_x.append(t)
		signal_y.append(y)
		I.I_V_j[i] = y

#### Monitors ####

# Check sinwave on input
I_M_s = SpikeMonitor(I)
I_M_V = StateMonitor(I, 'V', record=True)
I_M_V_j = StateMonitor(I, 'I_V_j', record=True)

#### Run Simulation ####
run (simtime)

#### Clean up and misc tasks ####
signal_x = array(signal_x)
signal_y = array(signal_y)

# Plotting
figure()
subplots_adjust(hspace=0.8)
subplots_adjust(wspace=0.5)

## First Column
# Input signal
subplot(321)
plot(signal_x / ms, signal_y / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Input Signal')
# Input neuron voltage sample
subplot(323)
plot(I_M_V.times / ms, I_M_V[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Input Neuron Voltage')
# Input group raster plot
subplot(325)
raster_plot(I_M_s, title='Input activity', newfigure=False)

## Second Column
# Neuron currents
subplot(322)
plot(L_M_ge.times / ms, L_M_ge[0] / mV)
plot(L_M_gi.times / ms, L_M_gi[0] / mV)
xlabel('Time (ms)')
ylabel('Currents (mV)')
title('Liquid Neuron Currents')
legend(('ge', 'gi'), 'upper right')
# Neuron voltage
subplot(324)
plot(L_M_V.times / ms, L_M_V[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Liquid Neuron Voltage')
# Liquid activety
subplot(326)
raster_plot(L_M_s, title='Liquid Activity', newfigure=False)
# Show
show()
