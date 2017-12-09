import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


def output_real_and_predict_data(y_real,y_pred,path,filename):
    """
    rt
    :param y_real: list
    :param y_pred: list
    :param path:
    :param filename:
    :return: None
    """
    with open(path+filename+'_real.csv','w') as fout:
        for val in y_real:
            fout.write(str(val)+',\n')
    with open(path + filename + '_predict.csv', 'w') as fout:
        for val in y_pred:
            fout.write(str(val) + ',\n')

    draw(path+filename+'_real.csv',path + filename + '_predict.csv',filename,path+filename+'.jpg')

    print "finish"+path+filename


def draw(realfilePath, predictfilePath, graphTitle, outputPath):
    tableReal = []
    for line in csv.reader(open(realfilePath)):
        tableReal.append(line[0])

    tablePred = []
    for line in csv.reader(open(predictfilePath)):
        tablePred.append(line[0])

    relativeError = []
    for i in range(len(tableReal)):
        if float(tableReal[i])!=0:
            relativeError.append(100 * abs(float(tablePred[i]) - float(tableReal[i])) / float(tableReal[i]))

    relativeError = sorted(relativeError, key=float)

    sampleRate = []
    length = len(relativeError)
    for i in range(length):
        sampleRate.append(100 * (i + 1) / length)

    _draw(relativeError, sampleRate, graphTitle, outputPath)


def _draw(xData, yData, title, outputPath):
    x = np.array(xData)
    y = np.array(yData)
    fig = plt.figure(1, (8, 5))
    ay = fig.add_subplot(1, 1, 1)
    plt.plot(x, y, linestyle='-', color='blue', marker='<')
    plt.grid(True, color='k', linestyle='--', linewidth='1')
    plt.title(title)
    plt.xlabel('relative prediction errors (%)')
    plt.ylabel('samples (%)')

    fmt = '%.2f%%'
    ticks = mtick.FormatStrFormatter(fmt)
    ay.yaxis.set_major_formatter(ticks)
    ay.xaxis.set_major_formatter(ticks)

    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 100])

    # plt.show()
    plt.savefig(outputPath)
    plt.close()