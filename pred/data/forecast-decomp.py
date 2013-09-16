import sys
import scipy as sp
import pylab as pl
import Oger
from M3 import *
import rpy2.robjects as robjects
r = robjects.r

r('''
	# Needed libs
	require(forecast)
	require(mgcv)

	# Data decomposition
	decomp <- function(x,transform=TRUE)
	{
	  # Transform series
	  if(transform & min(x,na.rm=TRUE) >= 0)
	  {
	    lambda <- BoxCox.lambda(na.contiguous(x))
	    x <- BoxCox(x,lambda)
	  }
	  else
	  {
	    lambda <- NULL
	    transform <- FALSE
	  }
	  # Seasonal data
	  if(frequency(x)>1)
	  {
	    x.stl <- stl(x,s.window="periodic",na.action=na.contiguous)
	    trend <- x.stl$time.series[,2]
	    season <- x.stl$time.series[,1]
	    remainder <- x - trend - season
	  }
	  else #Nonseasonal data
	  {
	    tt <- 1:length(x)
	    trend <- rep(NA,length(x))
	    trend[!is.na(x)] <- fitted(gam(x ~ s(tt)))
	    season <- NULL
	    remainder <- x - trend
	  }
	  return(list(x=x,trend=trend,season=season,remainder=remainder,transform=transform,lambda=lambda))
	}

	# Adjust data (deasonalise and detrend)
	adjust <- function(dec, freq)
	{
		if(freq > 1)
		{
			fits <- dec$trend + dec$season
		} else 
		{
			# Nonseasonal data
		    fits <- dec$trend
		}
		adj.x <- dec$x - fits + mean(dec$trend, na.rm=TRUE)

		# Backtransformation of adjusted data
		if(dec$transform)
		{
    		tadj.x <- InvBoxCox(adj.x,dec$lambda)
		} else
		{
		    tadj.x <- adj.x
		}
		return(tadj.x)
	}	

	# Undo adjustment
	undo_adjust <- function(x, freq, dec)
	{
		if(dec$transform)
		{
			adj.x <- BoxCox(x, dec$lambda)
		} else
		{
			adj.x <- x
		}

		if(freq >1)
		{
			fits <- dec$trend + dec$season
		} else
		{
			fits <- dec$trend
		}
		adj.x <- adj.x + fits - mean(dec$trend, na.rm=TRUE)
		
		if(dec$transform)
		{
			adj.x <- InvBoxCox(adj.x, dec$lambda)
		}
		return(adj.x)
	}

	# Normalise data into z-scores
	normalise <- function(x)
	{
		mean <- mean(x)
		stdev <- sd(x)
		x <- (x - mean)/stdev
		return(list(x=x, mean=mean, stdev=stdev))
	}

	undo_normalisation <- function(x, mean, stdev)
	{
		x <- (x * stdev) + mean
		return(x)
	}
''')


# SMAPE after Ahmed et al
def smape(x, y):
	m = len(x)
	sum = 0.0
	for i in range(0, m):
		err = sp.absolute(y[i] - x[i]) / ((sp.absolute(y[i]) + sp.absolute(x[i])) / 2.0)
		sum = sum + err
	sum = sum / m
	return sum


# Detrend, Deseasonalise and normalise
def preprocess(x, freq):
	ret = robjects.FloatVector(x)
	#print(r_x.r_repr())
	ret = r.ts(ret, f=freq)
	#print(r_ts.r_repr())
	decomp_x = r.decomp(ret)
	#print(decomp_x.r_repr())
	#ret = r.adjust(decomp_x, freq)
	#print(x_adj.rx())
	ret = r.normalise(ret)
	mean = ret.rx('mean')[0]
	stdev = ret.rx('stdev')[0]
	ret = sp.array(ret.rx('x'))[0]
	ret = sp.reshape(ret, (len(ret), 1))

	return (ret, mean, stdev, decomp_x)

def postprocess(x, freq, mean, stdev, decomp_x):
	r_x = robjects.FloatVector(x)
	ret = r.undo_normalisation(r_x, mean, stdev)
	#ret = r.undo_adjust(ret, freq, decomp_x)
	ret = sp.array(ret)
	ret = sp.reshape(ret, (len(ret), 1))

	return ret

def get_samples(x, window, n_samples):
	"""
	Take vector and return moving window subsequences as samples for validation
	on one-step-ahead prediction
	"""
	samples = []
	start = 0
	for i in range(0, n_samples):
		start = i
		end = start + window
		#print start, end
		sample = x[start:end, :]
		#print str(i) + " : " +str(sample)
		samples.append([sample])

	#print samples
	return samples


# Get the data
m3 = M3(sys.argv[1]).get_series()
# Select long monthly timeseries
series = [x for x in m3 if x.period == "MONTHLY" and x.n > 80]
series_index = int(sys.argv[2])
print "Found %d timeseries, selected series %d" % (len(series), series_index)
selected_series = series[series_index]
x = selected_series.data

# detrend, deseasonalise and normalise data
nx, mean, stdev, decomp = preprocess(x, 12)
#na = postprocess(nx, 12, mean, stdev, decomp)
#for i in range(0, len(x)):
#	print x[i], na[i]
#sys.exit(0)

# Prepare training and test sets
horizon = selected_series.nreq  # forecast horizon
train_size = selected_series.n - horizon
#train_size = 80 - horizon
window = int(0.5 * train_size)
washout = int(0.5 * window)
n_samples = train_size - window + 1
print "Forecast horizon: %d, Observations: %d, Window: %d, Washout: %d, Samples: %d" % (horizon, train_size, window, washout, n_samples)

# Create train and test sets
test_data = nx[-horizon:]
test_set = [test_data]
train_data = nx[:train_size]
train_set = get_samples(nx, window, n_samples)
n_folds = 5

# Reservoir size
size = int(sys.argv[3])

# Set the readout function
readout=Oger.nodes.RidgeRegressionNode()

# Igore the initial states of the reservoir
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode,washout)

# Do the grid search n times and print the results
for i in sp.arange(1):
	
	# Randomly shuffle train_set for cross validation
	sp.random.shuffle(train_set)
	
	# Create reservoir
	#reservoir=Oger.nodes.LeakyReservoirNode(output_dim=400,leak_rate=0.4,input_scaling=.05,bias_scaling=.2,reset_states=False)
	reservoir=Oger.nodes.LeakyReservoirNode(output_dim=size, reset_states=False)
	
	# Create the flow
	flow=Oger.nodes.FreerunFlow([reservoir,readout],freerun_steps=horizon)
	
	# Optimise
	# Heavy
	#gridsearch_parameters={readout:{'ridge_param':sp.logspace(-8,1,num=10)}, reservoir:{'output_dim':sp.arange(100, 1100, 100),'leak_rate':sp.logspace(-8, 0, num=10), 'input_scaling':sp.logspace(-8, 0, num=10)}}
	# Medium
	gridsearch_parameters={readout:{'ridge_param':sp.logspace(-8,1,num=10)}, reservoir:{'leak_rate':sp.logspace(-8, 0, num=10), 'input_scaling':sp.logspace(-8, 0, num=10)}}
	#gridsearch_parameters={readout:{'ridge_param':sp.logspace(-8,1,num=10)}, reservoir:{'leak_rate':sp.logspace(-8, 0, num=10)}}
	# Light
	#gridsearch_parameters={readout:{'ridge_param':sp.logspace(-8,1,num=5)}, reservoir:{'output_dim':[100],'leak_rate':sp.logspace(-8, 0, num=5), 'input_scaling':sp.logspace(-8, 0, num=5)}}
	# TEST ONLY
	#gridsearch_parameters={readout:{'ridge_param':10**sp.arange(-1,1,0.5)}, reservoir:{'output_dim':sp.arange(100, 300, 100)}}

	# Store search params
	plist = []
	for n in gridsearch_parameters[readout]:
		plist.append((readout, n))
	for n in gridsearch_parameters[reservoir]:
		plist.append((reservoir, n))
		
	#Instantiate an optimizer
	opt=Oger.evaluation.Optimizer(gridsearch_parameters,Oger.utils.nrmse)

	opt.grid_search([[],train_set],flow,cross_validate_function=Oger.evaluation.n_fold_random, n_folds=n_folds, progress=True)
		
	opt_flow=opt.get_optimal_flow(verbose=False)
	
	# Train the flow
	opt_flow.train([[], train_set])
	prediction = opt_flow.execute(nx)
	print prediction.shape
	adjusted = postprocess(prediction, 12, mean, stdev, decomp)
	#print Oger.utils.nrmse(x, adjusted)
	#print Oger.utils.rmse(x, adjusted)
	smape_err = smape(x, adjusted)
	
	opt_values=opt.get_minimal_error()

	if (i == 0):
		header = "SeriesIndex, ID, NRMSE, SMAPE"
		values = [series_index, selected_series.id, opt_values[0], smape_err[0]] 
		for n,p in plist:
			header = header + ", " + p
			values.append(opt_values[1][n][p])
		print header

	s = "" +  str(values[0]) + ", " + values[1]
	for i in range(2, len(values)):
		s = s + ", %0.5f" % values[i]
	print s
	

pl.plot(x, label='expected')
pl.plot(adjusted, label='forecast')
pl.legend(loc='upper right')
pl.show()

