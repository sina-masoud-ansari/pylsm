import Oger
import matplotlib.pyplot as plt
import numpy as np

signal_length = 5000
training_sample_length = int(0.7 * signal_length)
test_sample_length = signal_length - training_sample_length
freerun_steps = test_sample_length
print "Signal length: %d, train: %d, test %d" % (signal_length, training_sample_length, test_sample_length)

# create signal
signal = np.arange(0, signal_length)
signal = np.sin(0.1 * signal)
# partition into train and test
training_signal = signal[:training_sample_length]
test_signal = signal[training_sample_length:]

# create reservoir
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=400, leak_rate=0.4, input_scaling=0.05, bias_scaling=0.2, reset_states=False)
readout = Oger.nodes.RidgeRegressionNode()
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 500)

# setup flow
flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=freerun_steps)

# train readout
#training_signal = [[training_signal]]
#print training_signal
flow.train([[], [[training_signal]]])
result = flow.execute(training_signal)

# Plotting
plt.figure()
plt.plot(np.concatenate([training_signal, test_signal]))
plt.plot(np.concatenate([training_signal, result]))
plt.show()
