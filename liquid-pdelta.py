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
G_n = 1000
Ge_n = 700
Gi_n = 300
G = NeuronGroup(G_n, model=eqs, threshold=Vt, reset=Vr)
Ge = G.subgroup(Ge_n) # Excitatory neurons
Gi = G.subgroup(Gi_n)  # Inhibitory neurons

# Output Neuron Group
O_n = 100
O = NeuronGroup(O_n, model=eqs, threshold=Vt, reset=Vr)
#Gs = rnd.sample(xrange(4000), 1000) # Neurons representing state
#print Gs

# Connections
Ce = Connection(Ge, G, 'ge', sparseness=0.02, weight=we)
Ci = Connection(Gi, G, 'gi', sparseness=0.02, weight=wi)

# Connections to readout module
Co = Connection(G, O, 'V', structure='dense')
W = zeros((G_n, O_n))
# remember to add a 1 for the threshold / bg current to each weight vector
for i in range(0, G_n):
	for j in range(0, O_n):
		W[i][j] = np.random.random_sample() * nS
Co.connect(G, O, W) 
	
# Monitors
M = SpikeMonitor(G, record=True)
M_O = SpikeMonitor(O, record=True)
MV = StateMonitor(G, 'V', record=0)
Mge = StateMonitor(G, 'ge', record=0)
Mgi = StateMonitor(G, 'gi', record=0)
MPs = PopulationRateMonitor(O, 2.5 * ms)

# Init
G.V = Vr + (Vt - Vr) * rand(len(G))
O.V = Vr + (Vt - Vr) * rand(len(O))


# look like this needs to be online
def train(pdelta, W, state, rate, target):
	w = zeros((W.shape[1], W.shape[0]))
	for i in range(0, W.shape[0]):
		for j in range(0, W.shape[1]):
			w[j][i] = W.get_element(i, j)
	err = np.absolute(rate - target)
	alpha = 1.0
	z = alpha * np.ones(w.shape[1] + 1)
	thresh = np.zeros(w.shape[0])
	if err > pdelta.eps:
		for i in range(0, w.shape[0]):
			delta = None
			if (rate > (target + pdelta.eps) and (state[i] >= 0)):
				delta = -z
			elif (rate < (target - pdelta.eps) and (state[i] < 0)):
				delta = z
			elif (rate <= (target + pdelta.eps) and (0 <= state[i]) and (state[i] < pdelta.gamma)):
				delta = pdelta.mu * z
			elif (rate >= (target - pdelta.eps) and (-pdelta.gamma < state[i]) and (state[i] < 0)):
				delta = mu * -z
			else:
				delta = 0.0
			Wi = w[i, :]
			Wi = np.append(Wi, 1)
			Wi2 = np.dot(Wi, Wi)
			temp = Wi - eta * (Wi2 - 1.0) * Wi + pdelta.eta * delta
			thresh[i] = temp[-1]
			w[i, :] = temp[:-1]
		
			#for i in range(0, G_n):
			#for j in range(0, O_n):
			#W[i][j] =  np.random.random_sample() * 10 * uS
	return DenseConnectionMatrix(np.transpose(w)), thresh
	

def createSamples(input, target):
	ntrain = int(1.0 * len(input))
	if len(input) != len(target):
		print "Sample creation error: input and target lengths are not equal"
		sys.exit(1)
	samples = []
	for i in range(0, len(input)):
		samples.append((input[i], target[i]))
	return samples

# Operation
@network_operation(clock=defaultclock)
def update():
	t = defaultclock.t
	# get state
	window = 20 * ms
	state = ones(O_n) * -1.0
	for i in range(0, O_n):
		s = M_O[i]
		rev = s[::-1]
		#print t,i,rev 
		for ti in rev:
			#print ti
			if ti * second >= (t - window):
				#print i,ti
				state[i] = 1
			else:
				break
	# append 1 to state for threshold / background current
	state = np.append(state, -1)
	rate = net.pdelta.vote(state)
	#print rate, net.pdelta.squash(rate), net.pdelta.rho
	rate = net.pdelta.squash(rate)
	#print t, rate
	net.rates.append(rate)
	net.times.append(t)
	target = np.sin(20*2*np.pi*t)
	net.target.append(target)
	#print t, state
	# online training
	if (t < 0.5 * simtime):
		# train
		# create samples
		#samples = createSamples(net.rates, net.target)
		# may need to return a vector for the udpated thresholds / bg currents
		W, thresh = train(net.pdelta, Co.W, state, rate, target)
		O.Vt = thresh
		Co.W = W

# Network
#net = Network(G, Ge, Gi, O, Ce, Ci, Co, M, MV, Mge, Mgi, MPs, update)
net = MagicNetwork()

eps = 0.001 # some error > 0
#rho = 1.0 / (2.0 * eps) # resolution of squashing function
rho = O_n # resolution of squashing function
eta = 0.001 # the learning rate
gamma =  0.1 # clear margin for dot product (can be set by learning algorithm)
mu = 1.0 # importance of clear margin

net.pdelta = PDelta(rho, eps, eta, gamma, mu)
net.times = []
net.rates = []
net.target = []
net.trained = False

# Simulate
print "Training run"
net.run(0.5*simtime)
print "Test run"
net.run(0.5*simtime)

W = zeros((G_n, O_n))

# Plot
# First row
subplot(311)
raster_plot(M_O, title='Readout', newfigure=False)
# Second row
#subplot(323)
#plot(MV.times / ms, MV[0] / mV)
#xlabel('Time (ms)')
#ylabel('V (mV)')
#subplot(324)
#plot(Mge.times / ms, Mge[0] / mV)
#plot(Mgi.times / ms, Mgi[0] / mV)
#xlabel('Time (ms)')
#ylabel('ge and gi (mV)')
#legend(('ge', 'gi'), 'upper right')
# Thrid row
subplot(312)
raster_plot(M, title='Liquid', newfigure=False)
subplot(313)
plot(np.array(net.times) / ms, np.array(net.rates))
plot(np.array(net.times) / ms, np.array(net.target))
xlabel('Time (ms)')
ylabel('Rate')
show()
