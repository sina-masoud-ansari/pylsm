import sys
import scipy as sp
import pylab as pl
import Oger

# A time series class
class M3TimeSeries:
	def __init__(self, id, n, nreq, type, series_id, date, desc, data):
		self.id = id
		self.n = int(n)
		self.nreq = int(nreq)
		self.period, self.domain = type.split("/")
		self.series_id = series_id
		self.year, self.month = date.split()
		self.desc = desc
		self.data = sp.array(data).reshape((self.n, 1))
	
	def normalise(self):
		"""
		Normalise to [-1, 1]
		"""
		min = sp.amin(self.data)
		max = sp.amax(self.data)
		mean = (max + min) / 2.0
		range = max - min
		self.data = (self.data - mean) / (range / 2.0)

	def preprocess(self):
		self.normalise()


	def __str__(self):
		return "ID: %s, N: %d, Req: %d, Period: %s, Domain: %s, SID: %s, Year: %s, Month: %s, Desc: %s" % (self.id, self.n, self.nreq, self.period, self.domain, self.series_id, self.year, self.month, self.desc)
	

def getSamples(ts, h, k, ntrain):
	"""
	Take a time series and return samples for a freerun flow
	"""
	train = []
	test = []

	for i in range(0, ntrain):
			train.append([ts[:k + h + i, :]])
	for i in range(ntrain, len(ts) - k - h + 1):
			test.append([ts[:k + h + i, :]])

	return train, test



# Get the data
input = sys.argv[1]
series = []
with open(input,'r') as f:
	while True:
		line=f.readline()
		if not line: 
			break
		line = line.strip().replace("-", "")
		#print "'"+line+"'"
		id, n, nreq, type, sid = line.split()
		data = []
		#print id, n, nreq, type, sid
		d8 = int(n) / 8
		m8 = int(n) % 8
		nlines = d8 if m8 == 0 else d8 + 1
		for i in range(0, nlines):
			points = f.readline().strip().split()
			for p in points:
				data.append(float(p))
		#print data
		if len(data) != int(n):
			print "Something is wrong, expected %s, found %d points" % (n, len(data))
			sys.exit(0)
		date = f.readline().strip()
		desc = f.readline().strip()
		series.append(M3TimeSeries(id, n, nreq, type, sid, date, desc, data))

# Select long monthly timeseries
monthly = [x for x in series if x.period == "MONTHLY" and x.n > 100]
print "Selected %d timeseries" % len(monthly)

# Prepare timeseries
ts = monthly[0]
ts.preprocess()

# Prepare training and test sets
h = 15 # number fo steps ahead to forecast
k = 12 # initial requirement for forecast
remainder = ts.n - k - h
ntrain = int(0.7 * remainder)
print "N: %d, MinObs: %d, Steps: %d, nTrain: %d" % (ts.n, k, h, ntrain)
trainset, testset = getSamples(ts.data, h, k, ntrain)
#trainset = [trainset[len(trainset)-1], trainset[len(trainset)-2]]
#trainset = [x for x in ]
#print "Train"
#for i in trainset:
#	print len(i), i[0], i[len(i)-2], i[len(i)-1]
#
#print "Test"
#for i in testset:
#	print len(i), i[0], i[len(i)-2], i[len(i)-1]
#print ts.data[ts.n-1]

# Create reservoir
#reservoir=Oger.nodes.LeakyReservoirNode(output_dim=400,leak_rate=0.4,input_scaling=.05,bias_scaling=.2,reset_states=False)
reservoir=Oger.nodes.LeakyReservoirNode(output_dim=400,leak_rate=1.1,input_scaling=.05,bias_scaling=0.2,reset_states=False)

# Set the readout function
readout=Oger.nodes.RidgeRegressionNode(ridge_param=0.79432)
# Igore the initial states of the reservoir
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode,10)

# Create the flow
flow=Oger.nodes.FreerunFlow([reservoir,readout],freerun_steps=h)

# Optimise
#gridsearch_parameters={readout:{'ridge_param':10**sp.arange(-4,0,.3)}, reservoir:{'leak_rate':sp.arange(0.1,1.5,0.1)}}
gridsearch_parameters={reservoir:{'leak_rate':sp.arange(0.1,1.5,0.05)}}
#gridsearch_parameters={reservoir:{'leak_rate':[1.1]}}
	
#Instantiate an optimizer
opt=Oger.evaluation.Optimizer(gridsearch_parameters,Oger.utils.nrmse)
	
#Do the grid search
opt.grid_search([[],trainset],flow,cross_validate_function=Oger.evaluation.leave_one_out)
	
#opt_flow = flow
opt_flow=opt.get_optimal_flow(verbose=True)
	
# Train the flow
opt_flow.train([[], trainset])
test = testset[len(testset)-3][0]
prediction = opt_flow.execute(test)
print Oger.utils.nrmse(test, prediction)

pl.plot(test)
pl.plot(prediction)
pl.show()

#pl.plot(sp.concatenate((testset[len(testset)-1][0][-2*h:])))
#pl.plot(sp.concatenate((prediction[-2*h:])))
#pl.xlabel('Timestep')
#pl.legend(['Target signal','Predicted signal'])
#pl.axvline(pl.xlim()[1]-h+1,pl.ylim()[0],pl.ylim()[1],color='r')
#pl.show()

# Plotting

#pl.plot(ts.data)
#pl.title(ts.domain+" - "+ts.desc)
#pl.xlabel(ts.period)
#pl.show()

