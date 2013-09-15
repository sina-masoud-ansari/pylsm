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


# Detrend, Deseasonalise and normalise
def preprocess(x, freq):
	r_x = robjects.FloatVector(x)
	#print(r_x.r_repr())
	r_ts = r.ts(r_x, f=freq)
	#print(r_ts.r_repr())
	decomp_x = r.decomp(r_ts)
	#print(decomp_x.r_repr())
	x_adj = r.adjust(decomp_x, freq)
	#print(x_adj.rx())
	norm_x = r.normalise(x_adj)
	mean = norm_x.rx('mean')[0]
	stdev = norm_x.rx('stdev')[0]
	ret = sp.array(norm_x.rx('x'))[0]
	ret = sp.reshape(ret, (len(ret), 1))
	#print ret
	
	#print mean,stdev
	#ret = robjects.FloatVector(ret[0])
	#renorm_x = r.undo_normalisation(ret, mean, stdev)
	#ret = sp.array(renorm_x)
	#print(ret)
	return (ret, mean, stdev, decomp_x)

def postprocess(x, freq, mean, stdev, decomp_x):
	r_x = robjects.FloatVector(x)
	ret = r.undo_normalisation(r_x, mean, stdev)
	ret = r.undo_adjust(ret, freq, decomp_x)
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
selected_index = 0
print "Found %d timeseries, selected series %d" % (len(series), selected_index)
selected_series = series[0]
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
window = int(0.4 * train_size)
washout = int(0.5 * window)
n_samples = train_size - window + 1
print "Forecast horizon: %d, Observations: %d, Window: %d, Washout: %d, Samples: %d" % (horizon, train_size, window, washout, n_samples)

# Create train and test sets
test_data = nx[-horizon:]
test_set = [test_data]
train_data = nx[:train_size]
train_set = get_samples(nx, window, n_samples)
n_folds = 5

# Randomly shuffle train_set for cross validation
# TODO: set random seed from jobid
sp.random.shuffle(train_set)
#print train_set

# Create reservoir
#reservoir=Oger.nodes.LeakyReservoirNode(output_dim=400,leak_rate=0.4,input_scaling=.05,bias_scaling=.2,reset_states=False)
reservoir=Oger.nodes.LeakyReservoirNode(output_dim=400,leak_rate=0.5,input_scaling=.05,bias_scaling=0.2,reset_states=False)

# Set the readout function
readout=Oger.nodes.RidgeRegressionNode(ridge_param=0.79432)
# Igore the initial states of the reservoir
Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode,washout)

# Create the flow
flow=Oger.nodes.FreerunFlow([reservoir,readout],freerun_steps=horizon)

# Optimise
#gridsearch_parameters={readout:{'ridge_param':10**sp.arange(-4,0,.3)}, reservoir:{'leak_rate':sp.arange(0.1,1.5,0.1)}}
gridsearch_parameters={readout:{'ridge_param':10**sp.arange(-4,0,.3)}}
#gridsearch_parameters={readout:{'ridge_param':10**sp.arange(-4,0,.3)}, reservoir:{'leak_rate':sp.arange(0.1,1.5,0.1), '_instance':range(5)}}
#gridsearch_parameters={reservoir:{'leak_rate':sp.arange(0.1,1.5,0.05)}}
#gridsearch_parameters={reservoir:{'leak_rate':[1.1]}}
	
#Instantiate an optimizer
opt=Oger.evaluation.Optimizer(gridsearch_parameters,Oger.utils.nrmse)
	
#Do the grid search
#opt.grid_search([[],trainset],flow,cross_validate_function=Oger.evaluation.leave_one_out)
opt.grid_search([[],train_set],flow,cross_validate_function=Oger.evaluation.n_fold_random, n_folds=n_folds)
	
#opt_flow = flow
opt_flow=opt.get_optimal_flow(verbose=True)
opt_values=opt.get_minimal_error()
print opt_values

# Train the flow
opt_flow.train([[], train_set])
#test = testset[len(test_set)-3][0]
prediction = opt_flow.execute(train_data)
adjusted = postprocess(prediction, 12, mean, stdev, decomp)
print Oger.utils.nrmse(x, adjusted)


pl.plot(x, label='expected')
pl.plot(adjusted, label='forecast')
pl.legend(loc='upper right')
pl.show()

