"""
Going to try just generating a population rate

"""

# using the CUBA model

import sys
import random as rnd
from brian import *
import numpy as np
from PDelta import PDelta

taum = 20 * ms          # membrane time constant
taue = 5 * ms          # excitatory synaptic time constant
taui = 10 * ms          # inhibitory synaptic time constant
Vt = -50 * mV          # spike threshold
Vr = -60 * mV          # reset value
El = -49 * mV          # resting potential
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight
wi = (20 * 4.5 / 10) * mV # inhibitory synaptic weight


# Sim params
np.random.seed(123)
defaultclock.dt = 1 * ms
simtime = 500 * ms


eqs = Equations('''
	dV/dt  = (ge-gi-(V-El))/taum : volt
	dge/dt = -ge/taue            : volt
	dgi/dt = -gi/taui            : volt
''')

# Neurons
G_n = 10
Ge_n = 7
Gi_n = 3
G = NeuronGroup(G_n, model=eqs, threshold=Vt, reset=Vr)
Ge = G.subgroup(Ge_n) # Excitatory neurons
Gi = G.subgroup(Gi_n)  # Inhibitory neurons

# Output Neuron Group
O = NeuronGroup(100, model=eqs, threshold=Vt, reset=Vr)
#Gs = rnd.sample(xrange(4000), 1000) # Neurons representing state
#print Gs

# Connections
Ce = Connection(Ge, G, 'ge', sparseness=0.02, weight=we)
Ci = Connection(Gi, G, 'gi', sparseness=0.02, weight=wi)

#Co = Connection(G, O, 'V', sparseness=0.1, weight=we)
Co = Connection(G, O, 'V', sparseness=0.1, weight=we)
#Co = Connection(G, O, 'V')
#alpha = 100 * nS
#for i in range(0, len(Gs)):
#	#print (Gs[i], i)
#	Co[Gs[i], i] = alpha

# Monitors
M = SpikeMonitor(G, record=True)
MV = StateMonitor(G, 'V', record=0)
Mge = StateMonitor(G, 'ge', record=0)
Mgi = StateMonitor(G, 'gi', record=0)
MPs = PopulationRateMonitor(O, 2.5 * ms)

# Init
G.V = Vr + (Vt - Vr) * rand(len(G))

# Operation
@network_operation(clock=defaultclock)
def update():
	t = defaultclock.t
	# get state
	window = 20 * ms
	state = ones(G_n) * -1.0
	for i in range(0, G_n):
		s = M[i]
		rev = s[::-1]
		#print t,i,rev 
		for ti in rev:
			#print ti
			if ti * second >= (t - window):
				#print i,ti
				state[i] = 1
			else:
				break
	rate = net.pdelta.vote(state)
	#print t, rate
	net.rates.append(rate)
	net.times.append(t)
	#print t, state
	if (t > 0.5 * simtime) and not net.trained:
		pass
		# train
		#net.trained = True
		#print "Trained at :" + str(t / ms)

# Network
#net = Network(G, Ge, Gi, O, Ce, Ci, Co, M, MV, Mge, Mgi, MPs, update)
net = MagicNetwork()

eps = 0.1 # some error > 0
rho = 1.0 / (2.0 * eps) # resolution of squashing function
eta = 0.001 # the learning rate
gamma =  0.1 # clear margin for dot product (can be set by learning algorithm)
mu = 1.0 # importance of clear margin

net.pdelta = PDelta(eps, rho, eta, gamma, mu)
net.times = []
net.rates = []
net.trained = False

# Simulate
net.run(simtime)

# Plot
# First row
subplot(311)
raster_plot(M, title='The CUBA network', newfigure=False)
# Second row
subplot(323)
plot(MV.times / ms, MV[0] / mV)
xlabel('Time (ms)')
ylabel('V (mV)')
subplot(324)
plot(Mge.times / ms, Mge[0] / mV)
plot(Mgi.times / ms, Mgi[0] / mV)
xlabel('Time (ms)')
ylabel('ge and gi (mV)')
legend(('ge', 'gi'), 'upper right')
# Thrid row
subplot(313)
plot(np.array(net.times) / ms, np.array(net.rates))
xlabel('Time (ms)')
ylabel('Rate')
show()
