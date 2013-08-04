"""
Class for implementing the p-delta learning rule in Auer et. al. 2002

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Create some test signal
np.random.seed(123)
t = np.linspace(0.0, 500.0, 1000) # time in milliseconds with dt = 0.1
f = 50
w = 2 * np.pi * f
y = np.sin(w * t)

ni = 16 # input vector size
nl = 64 # num nodes in layer
no = 1 # output vector size
sz = ni + no # sample vector size (input, target)

# Create training and test cases
nsamples = y.shape[0] / sz
samples = []
for i in range(0, nsamples):
	s = y[i*sz: (i+1)*sz]
	input = s[:sz-1]
	target = s[sz-1:]
	samples.append((input, target))
	#print input, target
ntrain = int(0.7 * nsamples)
ntest = nsamples - ntrain
training_set = samples[:ntrain]
test_set = samples[ntrain:]
print "nsamples: "+str(len(samples)) 
print "ntrain: "+str(len(training_set)) 
print "ntest: "+str(len(test_set)) 

# create matrix of weights from gaussian distribution
mu, sigma = 0, 0.1
W = np.random.normal(mu, sigma, (nl, ni + 1)) # the weights (+1 for threshold) 

# Training parameters
epochs = 20
eps = 0.01 # some error > 0
rho = 1.0 / (2.0 * eps) # resolution of squashing function
eta = 0.001 # the learning rate
gamma = 0.01 # clear margin for dot product (can be set by learning algorithm)
mu = 1.0 # importance of clear margin

# Incremental Training
for e in range(epochs):
	errsum = 0.0
	for tz,o in training_set:
		# add value for threshold (corresponds to weight ni + 1)
		z = np.append(tz, -1)
		#print "input: "+str(z)
		#print "target: "+str(o)
		# compute input value to perceptron
		v = np.dot(W, z)
		#print "v (z.w): "+ str(v)
		# map input to {-1, 1}
		fz = np.zeros(v.shape[0])
		for i in range(0, len(fz)):
			if (v[i] >= 0.0):
				fz[i] = 1
			else:
				fz[i] = -1
		#print "fz: "+str(fz)
		# Take the majority vote
		p = np.dot(fz, np.ones(fz.shape[0]))
		# squash the vote (p) to [-1, 1]
		sp = 0
		if (p < -rho):
			sp = -1.0
		elif (-rho <= p) and (p <= rho):
			sp = float(p / rho)
		else:
			sp = 1.0
		#print "rho: "+str(rho)+", p:"+ str(p) +", sp: "+str(sp)
		err = np.absolute(o - sp)
		#print "error: "+str(err)
		errsum = errsum + err
		# update weights if err > eps
		if (err > eps):
			delta = None
			# for each node, update the weight vectors
			for i in range(0, nl):
				#print "W["+str(i)+"]:"+str(W[i, :])
				#print "v["+str(i)+"]:"+str(v[i])
				# conditional update
				if (sp > (o + eps)) and (v[i] >= 0):
					delta = -z
				elif (sp < (o - eps)) and (v[i] < 0):
					delta = z
				elif (sp <= (o + eps)) and (0 <= v[i]) and (v[i] < gamma):
					delta = mu * z
				elif (sp >= (o - eps)) and (-gamma < v[i]) and (v[i] < 0):
					delta = mu * -z
				else:
					delta = np.zeros(z.shape[0])
				#print "delta: "+str(delta)
				# update weights for node i
				Wi = W[i, :]
				Wi2 = np.dot(Wi, Wi)
				#print "Wi.Wi: "+str(Wi2)
				W[i, :] = Wi - eta * (Wi2 - 1) * Wi + eta * delta
				#print "W_updated["+str(i)+"]:"+str(W[i, :])
	print "Epoch: "+str(e) + ", error: "+str(errsum / ntrain)

# Evaluate on test set
actual = []
result = []
for tz,o in test_set:
	z = np.append(tz, -1)
	# compute input value to perceptron
	v = np.dot(W, z)
	fz = np.zeros(v.shape[0])
	for i in range(0, len(fz)):
		if (v[i] >= 0.0):
			fz[i] = 1
		else:
			fz[i] = -1
	# Take the majority vote
	p = np.dot(fz, np.ones(fz.shape[0]))
	# squash the vote (p) to [-1, 1]
	sp = 0
	if (p < -rho):
		sp = -1.0
	elif (-rho <= p) and (p <= rho):
		sp = float(p / rho)
	else:
		sp = 1.0
	#print "rho: "+str(rho)+", p:"+ str(p) +", sp: "+str(sp)
	actual.append(sp)
	result.append(o)

# Plotting
plt.subplots_adjust(hspace=0.8)
plt.subplot(311)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Input signal')
plt.subplot(312)
plt.bar(range(W.shape[1]), W[0, :]) # plot weights for node 0
plt.xlabel('Weight Index')
plt.ylabel('Value')
plt.title('Sample Weights (Node 0)')
plt.subplot(313)
x = range(len(actual))
plt.plot(np.array(x), np.array(actual))
plt.plot(np.array(x), np.array(result))
plt.xlabel('Sample')
plt.ylabel('y(y)')
plt.title('test')
plt.show()

