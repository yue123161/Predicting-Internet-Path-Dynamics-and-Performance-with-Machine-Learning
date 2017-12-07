import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def main(realfilePath,predictfilePath,graphTitle):
	tableReal = []
	for line in csv.reader(open(realfilePath)):
		tableReal.append(line[0])
	
	tablePred = []
	for line in csv.reader(open(predictfilePath)):
		tablePred.append(line[0])
	
	relativeError = []
	for i in range(len(tableReal)):
		relativeError.append(100*abs(float(tablePred[i]) - float(tableReal[i]))/float(tableReal[i]))

	relativeError = sorted(relativeError, key = float)
	
	sampleRate = []
	length = len(relativeError)
	for i in range(length):  
		sampleRate.append(100*(i+1)/length) 


	draw(relativeError,sampleRate,graphTitle)

def draw(xData, yData,title):
	x = np.array(xData)
	y = np.array(yData)
	fig = plt.figure(1, (8,5))
	ay = fig.add_subplot(1,1,1)
	plt.plot(x,y,linestyle='-',color='blue',marker='<')
	plt.grid(True,color='k',linestyle='--',linewidth='1')
	plt.title(title)
	plt.xlabel('relative prediction errors (%)')
	plt.ylabel('samples (%)')

	fmt='%.2f%%'
	ticks = mtick.FormatStrFormatter(fmt)
	ay.yaxis.set_major_formatter(ticks)
	ay.xaxis.set_major_formatter(ticks)

	axes = plt.gca()
	axes.set_xlim([0,100])
	axes.set_ylim([0,100])
	
	plt.show()


if __name__ == '__main__':
	realfilePath = 'real.csv'#change here
	predictfilePath = 'predict.csv'#change here 
	graphTitle = 'this is title'#change here
	main(realfilePath,predictfilePath,graphTitle)

