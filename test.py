"""
Test the use of Brian to create LSM reservoir and train to reproduce sin wave
"""

from brian import *
from ModelSpace import Point3D, Cuboid3D 
from numpy import *
import math
import sys

### Simulation Parameters

dt = 2 * ms
defaultclock.dt = dt # default timestep is 0.1 ms
simtime = 1000 * ms 
nsteps = simtime / dt
ptrain = 0.7 # percentage of examples to use for training

### NOTES ####

# - Input neurons have no internal connections
# - Input neurons have no random initial voltage
# - Input neurons have no refractory period

#### Liquid ####

# Neuron parameters
# http://www.neurdon.com/2011/01/19/neural-modeling-with-python-part-1/
# http://briansimulator.org/docs/tutorial_2c_the_cuba_network.html
L_tau_m = 80 * ms	# liquid membrane time constant (R_m * C_m)
L_tau_e = 30 * ms	# liquid excitatory synaptic time constant
L_tau_i = 20 * ms	# liquid inhibitory synaptic time constant
L_V_t = 30 * mV	# liquid spike threshold
L_V_eq = 29 * mV	# liquid equilibrium potential (I * R_m)
L_V_r = 0 * mV	# liquid spike reset
L_w_e = 3 * mV	# excitatory synaptic weight
L_w_i =  3 * mV		# inhibitory synaptic weight

# Neuron equations (LIF)
L_eqs = Equations("""
	dV/dt = (-V + L_V_eq + ge - gi) / L_tau_m : volt
	dge/dt = -ge/L_tau_e : volt
	dgi/dt = -gi/L_tau_i : volt
""")

## Neuron equations (LIF)
#L_eqs = Equations("""
#	dV/dt = (-V + L_V_eq ) / L_tau_m : volt
#""")


# Liquid dimensions
L_x = 200	# X dimension
L_y = 1	# Y dimension
L_z = 1	# Z dimension
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
L_C_e = Connection(L_e, L, 'ge', sparseness=0.2, weight=L_w_e)
L_C_i = Connection(L_i, L, 'gi', sparseness=0.2, weight=L_w_i)


# Monitors
L_M_s = SpikeMonitor(L)
L_M_V = StateMonitor(L, 'V', record=True)
L_M_ge = StateMonitor(L, 'ge', record=True)
L_M_gi = StateMonitor(L, 'gi', record=True)

##### Input ####

# Poisson Source
#P_psp = 5.5 * mV
#P = PoissonGroup(10,rates=10*Hz)
#Connection(P, L, sparesness=0.2, weight=P_psp)
#P_M_s = SpikeMonitor(P, record=True)

# Neurons Parameters
I_tau_m = 20 * ms	# input membrane time constant
I_V_t = -50 * mV	# input spike threshold
I_V_r = -60 * mV		# input spike reset
I_V_eq = -51 * mV		# input rest potential
I_psp = 1 * mV	# input neuron post-synaptic potential

# I_V_j is the injection voltage from the signal source
I_eqs = Equations("""
	dV/dt = ( -V + I_V_j + I_V_eq) / I_tau_m : volt
	I_V_j : volt
""")

# Input dimensions
I_x = 10	# X dimension
I_y = 1	# Y dimension
I_z = 1	# Z dimension
I_n = I_x * I_y * I_z		# number of input neurons
I_offset = Point3D(L_x-2,0,0) 	# place input neurons next to liquid 
I_mspace = Cuboid3D(I_x, I_y, I_z, offset=I_offset)

# Define neuron group
I = NeuronGroup(I_n, model=I_eqs, threshold=I_V_t, reset=I_V_r)

# Assign random initial voltage
I.V = I_V_r + rand(I_n) * (I_V_t - I_V_r)

# Input neuron group connections
# -- currently no interal connections
# May want to create inhibitory and excitatory currents from input to liquid
I_C_L = Connection(I, L, 'V', sparseness=0.5, weight=lambda: rand() * mV)


# Signal parameters (Constant sinewave as input)
A = 4 * mV 			# amplitude
c = 6 * mV				# vertical shift
f = 5 * hertz			# frequency
w = 2 * math.pi * f

# Signal generator
signal_x = [] # x values
signal_y = [] # y values
@network_operation(clock=defaultclock)
def updateInput():
	"""Sine wave"""
	t = defaultclock.t # (ms)
	#phi = (float(i) / 7) * 2 * math.pi # phase
	phi = 0
	y = A * sin(w*t + phi) + c # (mV)	
	#y = 2 * mV
	signal_x.append(t)
	signal_y.append(y)
	I.I_V_j = y
	#for i in range(0, len(I.V)):

#### Monitors ####

# Check sinwave on input
I_M_s = SpikeMonitor(I)
I_M_V = StateMonitor(I, 'V', record=True)
I_M_V_j = StateMonitor(I, 'I_V_j', record=True)


#### Output neurons ####

# Neuron parameters (Linear Unit)
#O_tau_m = 1 * ms
#O_psp = 0.25 * mV
#O_eqs = Equations("""
#	dV/dt = -V / O_tau_m : volt
#""")

# Neuron parameters (Linear Unit)
O_tau_m = 1 * ms
O_eqs = Equations("""
	V : volt
""")


# Ouput dimensions
O_x = 1	# X dimension
O_y = 1	# Y dimension
O_z = 1	# Z dimension
O_n = O_x * O_y * O_z		# number of input neurons
O_offset = Point3D(L_x + 1,0,0) 	# place input neurons next to liquid 
O_mspace = Cuboid3D(O_x, O_y, O_z, offset=O_offset)

# Define neuron group
O = NeuronGroup(O_n, model=O_eqs)

# Define connections
W = rand(len(L), len(O)) * nS
#print W
O_C_L = Connection(L, O, 'V')
O_C_L.connect(L, O, W)

# Setup Monitors
O_M_V = StateMonitor(O, 'V', record=True)

#### Run Simulation ####
run (simtime)

#### Clean up and misc tasks ####
print len(signal_x)
print len(signal_y)
x = array(signal_x)
y = array(signal_y)

#for i in range(0, len(signal_x)):
#	print (signal_x[i], signal_y[i])

#### Learning ####

# Create Liquid state matrix S
S = zeros( (nsteps, L_n) )
for n in  L_M_s.spiketimes:
	for t in L_M_s.spiketimes[n]:
		t = t * second / dt
		S[t][n] = 1
		
#for s in S:
#	print s
print "S orig. " + str(S.shape)
print "y orig. " + str(y.shape)
print "x orig. " + str(x.shape)
# match signal input length with state output length
trim = int(0.2 * nsteps)
y = delete(y, s_[S.shape[0]:], 0)
x = delete(x, s_[S.shape[0]:], 0)
print "y clamp. " + str(y.shape)
print "x clamp. " + str(x.shape)
# Remove first x ms (trim timesteps)
S = delete(S, s_[:trim], 0)
print "S trim. " + str(S.shape)
y = delete(y, s_[:trim], 0)
x = delete(x, s_[:trim], 0)
print "Y trim " + str(y.shape)
# Split into training and test sets
ntrain = S.shape[0] * ptrain
#ntest = S.shape[0] - ntrain
S_train = S[:ntrain, :]
S_test = S[ntrain:, :]
print "S_train " + str(S_train.shape)
print "S_test " + str(S_test.shape)
y_train = y[:ntrain]
y_test = y[ntrain:]
x_train = x[:ntrain]
x_test = x[ntrain:]
print "y_train " + str(y_train.shape)
print "x_train " + str(x_train.shape)
print "y_test " + str(y_test.shape)
print "x_test " + str(x_test.shape)

# Solve linear system
result = linalg.lstsq(S_train, y_train)
W_train = result[0]
error = result[1]
print "Error: " + str(error)
print "W_train: " + str(W_train.shape)
#W_train = W_train.reshape(W_train.shape[0], 1)

# Plot first set:

figure(1)
subplots_adjust(hspace=0.8)
subplots_adjust(wspace=0.5)

## First Column (Input Neurons)
# Input signal
subplot(331)
y = array(signal_y)
x = array(signal_x)
plot(x / ms, y / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Input Signal')
# Input neuron voltage sample
subplot(334)
plot(I_M_V.times / ms, I_M_V[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Input Neuron Voltage')
# Input group raster plot
subplot(337)
raster_plot(I_M_s, title='Input activity', newfigure=False)

## Second Column (Liquid Neurons)
# Neuron currents
subplot(332)
plot(L_M_ge.times / ms, L_M_ge[0] / mV)
plot(L_M_gi.times / ms, L_M_gi[0] / mV)
xlabel('Time (ms)')
ylabel('Currents (mV)')
title('Liquid Neuron Currents')
legend(('ge', 'gi'), 'upper right')
# Neuron voltage
subplot(335)
plot(L_M_V.times / ms, L_M_V[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Liquid Neuron Voltage')
# Liquid activety
subplot(338)
raster_plot(L_M_s, title='Liquid Activity', newfigure=False)

# Set the output weights and run again
#for i in range(0, O_C_L.W.shape[0]):
#	for j in range(0, O_C_L.W.shape[1]):
#		O_C_L.W.set_element(i, j, W_train[i][j])
#		#print W[i][j], O_C_L.W[i][j] 

#signal_y = []
#signal_x = []
#reinit(states=False)
#run (simtime)

# Create Liquid state matrix S
#S = zeros( (nsteps, L_n) )
#for n in  L_M_s.spiketimes:
#	for t in L_M_s.spiketimes[n]:
#		t = t * second / dt
#		S[t][n] = 1

#print "S_new: "+str(S.shape)

#S = delete(S, s_[:trim], 0)
res = dot(S_test, W_train)
print "S_test: " + str(S_test.shape)+" x W_train: " + str(W_train.shape)+" = r: "+str(res.shape)

## Third Column (Output Neurons)
# Test signal
subplot(333)
plot(O_M_V.times / ms, O_M_V[0] / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Output Voltage')

# 
subplot(336)
plot(x_test / ms, y_test / mV)
plot(x_test / ms, res / mV)
xlabel('Time (ms)')
ylabel('Value)')
title('Learned Output')
#legend(('y\'', 'y'), 'upper right')

# Show
show()
