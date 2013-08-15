"""
Attemp at a classification pipeline
"""

"""
Start with sequences and jitter

"""

import sys
import random as rnd
import math
from brian import *
import numpy as np
from PDelta import PDelta

def generatePossionSpikeTrain(rate, channels, period):
	dt = defaultclock.dt
	t = 0 * dt
	spikes= []
	r = rate * dt
	for i in range(0, int(period / dt)):
		t = t + dt
		for j in range(0, channels):
			x = np.random.uniform()
			if r > x:
				spikes.append((j, t))
	return spikes

def addNoise(spikes, var):
	mu = 0
	sigma = sqrt(var) 
	shifted = []
	for n,t in spikes:
		t0 = t
		t = t + rnd.gauss(mu, sigma) * ms
		# filter out negative times
		if t > 0:
			#print (n,t0),(n, t)
			shifted.append((n,t))
	# sort by time
	shifted = sorted(shifted, key=lambda spike : spike[1])
	#print spikes[0], shifted[0]
	return shifted

def train(net, targets):
	# train each readout to match its target for this given state sequence
	for r in range(0, len(rconn)):
		print "training readout" + str(r) + " with target "+str(targets[r])
		# for each state
		for t in range(0, len(net.rtimes)):
			time = net.rtimes[t]
			output = net.rrates[r][t]
			target = targets[r]
			print "training for time: "+str(time)	
			print "output is " + str(output) + ", target is :"+str(target)
			error = fabs(output - target)
			print "error is : "+str(error)
			W = np.transpose(rconn[r].W)
			if error > pdelta.eps:

			

# Sim params
# ---------------------------------------------------
seed = 123
rnd.seed(seed)
np.random.seed(seed)
defaultclock.dt = 1 * ms
simtime = 100 * ms
neurons = []
connections = []
monitors = []
methods = []
#----------------------------------------------------

# Neuron params #
#----------------------------------------------------------
taum = 20 * ms          # membrane time constant
taue = 5 * ms          # excitatory synaptic time constant
taui = 10 * ms          # inhibitory synaptic time constant
Vt = 10 * mV          # spike threshold
Vr = 0 * mV          # reset value
El = 9 * mV          # resting potential
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight
wi = (20 * 4.5 / 10) * mV # inhibitory synaptic weight

G_eqs = Equations('''
	dV/dt  = (-V + El + ge - gi)/taum : volt
	dge/dt = -ge/taue : volt
	dgi/dt = -gi/taui : volt
''')

#-----------------------------------------------------------

# Liquid Neurons
G_n = 200
Ge_n = 140
Gi_n = 60
G = NeuronGroup(G_n, model=G_eqs, threshold=Vt, reset=Vr)
neurons.append(G)
C_G = Connection(G, G, 'ge', sparseness=0.02, weight=we)
connections.append(C_G)
#Ge = G.subgroup(Ge_n) # Excitatory neurons
#neurons.append(Ge)
#Gi = G.subgroup(Gi_n)  # Inhibitory neurons
#neurons.append(Gi)
#Ce = Connection(Ge, G, 'ge', sparseness=0.02, weight=we)
#connections.append(Ce)
#Ci = Connection(Gi, G, 'gi', sparseness=0.02, weight=wi)
#connections.append(Ci)

# Init
G.V = Vr #Vr + (Vt - Vr) * rand(len(G))

M_G = SpikeMonitor(G, record=True)
monitors.append(M_G)
M_G_V = StateMonitor(G, 'V', record=0)
monitors.append(M_G_V)
M_G_ge = StateMonitor(G, 'ge', record=0)
monitors.append(M_G_ge)
M_G_gi = StateMonitor(G, 'ge', record=0)
monitors.append(M_G_gi)
##-------------------------------------------------------------##
# Input 

# 2D array of pattern templates and their jittered variants
channels = 40
variance = 32 # ms
npatterns = 1
nvariants = 2
patterns = []
for i in range(0, npatterns):
	# generate templane
	spikes = generatePossionSpikeTrain(100*Hz, channels, simtime)
	# create variants of the template
	variants = []
	for j in range(0, nvariants):
		variants.append(addNoise(spikes, variance))
	patterns.append(variants)

P = None
M_P = None
ntrain = max(1, int(0.7 * nvariants))

# build connection lists from input to liquid
# each channel connects randomly to some number of liquid neurons
# according to a normal distribution
conn = []
psp = 1 * mV
for i in range(0, channels):
	sample = rnd.sample(xrange(G_n), int(0.02* G_n))
	for j in sample:
		conn.append((i, j))

# Readouts
R_tau_m = 20 * ms
R_El = 9 * mV
R_Vt = 10 * mV
R_Vr = 0 * mV
R_psp = 1 * mV
R_eqs = Equations('''
	dV/dt = (-V + R_El)/R_tau_m : volt
''')
readouts = []
rconn = []
rmon = []
rsize = 50
R_w = 0.5 * msiemens
for n in range(0, npatterns):
	R = NeuronGroup(rsize, model=R_eqs, reset=R_Vr, threshold=R_Vt)
	C = Connection(G, R, 'V', structure='dense')
	W = np.zeros((G_n, rsize))
	for i in range(0, W.shape[0]):
		for j in range(0, W.shape[1]):
			W[i][j] = np.random.random_sample() * R_w
	print W
	C.connect(G, R, W)
	M_R = SpikeMonitor(R, record=True)
	readouts.append(R)
	rconn.append(C)
	rmon.append(M_R)


@network_operation(clock=defaultclock)
def update():
	t = defaultclock.t
	window = 10 * ms
	state = zeros(G_n)
	# collect the liquid states
	for i in range(0, G_n):
		s = M_G[i]
		rev = s[::-1]
		for ti in rev:
			if ti * second >= (t - window):
				state[i] = 1
			break
	net.states.append(state)
	# find corresponding readout states
	net.rtimes.append(t)
	for r in range(0, len(net.rrates)):
		rstate = np.ones(rsize) * -1
		monitor = rmon[r]
		for i in range(0, rsize):
			s = monitor[i]
			rev = s[::-1]
			#print rev
			for ti in rev:
				#print ti * second
				if ti * second >= (t - window):
					#print t, i, ti*second
					rstate[i] = 1
				break
		# find rate from state
		rate = net.pdelta.vote(rstate)	
		rate = net.pdelta.squash(rate)
		net.rrates[r].append(rate)
		#print t, rate
	#print t, [s for s in state if s == 1]	

methods.append(update)
# train readouts
epochs = 1
rho = rsize
eps = 0.001
eta = 0.001
gamma = 0.1
mu = 1.0
pdelta = PDelta(rho, eps, eta, gamma, mu)
for e in range(0, epochs):
	count = 0
	for p in range(0, npatterns):
		pattern = patterns[p]
		y = np.zeros(npatterns)
		y[p] = 1
		for v in range(0, len(pattern)):
			variant = pattern[v]
			# create generator group
			P = SpikeGeneratorGroup(channels, variant)
			# Connect it to liquid
			C = Connection(P, G, 'V')
			for c1,c2 in conn:
				C[c1,c2] = psp
			M_P = SpikeMonitor(P, record=True)
			#present pattern and train network
			#reinit(states=True)
			net = Network(neurons, methods, connections, readouts, rconn, rmon, monitors, P, C, M_P)
			net.reinit(states=True)
			net.states = []
			net.rrates = [] # readout rates 
			for r in range(0, len(rconn)):
				net.rrates.append([])
			net.rtimes = []
			net.pdelta = pdelta
			# run and collect state results
			net.run(simtime)
			# plot history
			figure(count)
			subplots_adjust(hspace=2.0)
			subplots_adjust(wspace=0.5)
			subplot(411)
			raster_plot(M_P, title='Pattern', newfigure=False)
			subplot(412)
			raster_plot(M_G, title='Liquid', newfigure=False)
			# perform training
			train(net, y)
			subplot(413)
			raster_plot(rmon[0], title='Readout 0', newfigure=False)
			subplot(414)
			plot(np.array(net.rtimes) / ms, np.array(net.rrates[0]))
			#plot(np.array(net.rtimes) / ms, np.array(net.rrates[1]))
			title("Readouts")
			xlabel("Time (ms)")
			title("Rate")
			#subplot(414)
			#plot(M_G_ge.times / ms, M_G_ge[0] / mV)
			#plot(M_G_ge.times / ms, M_G_gi[0] / mV)
			#title("Neuron: ge and gi")
			#xlabel("Time (ms)")
			#title("Voltage (mV)")
	
			count = count+1
		
# run tests

#		- this involves running and restarting the sim

show()

##-------------------------------------------------------------##


# Maigic

# Run


# Plot

