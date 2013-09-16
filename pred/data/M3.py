import scipy as sp

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
	

# Loads an M3 data set
class M3:
	def __init__(self, fname):
				
		# Get the data
		self.series = []
		with open(fname,'r') as f:
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
					print "Something is wrong, expected %s, found %d points in '%s'" % (n, len(data), fname)
					sys.exit(0)
				date = f.readline().strip()
				desc = f.readline().strip()
				self.series.append(M3TimeSeries(id, n, nreq, type, sid, date, desc, data))

	def get_series(self):
		return self.series
