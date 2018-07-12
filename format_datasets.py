import sys, numpy

def writeToFile(filename,x,y):
    with open(filename,'w') as f:
        for i in range(len(x)):
            for j in x[i]:
                f.write(str(j)+', ')
            f.write(str(y[i])+'\n')

def exportData(filename,data):
    filename = filename.split('.')[0]

    header = []
    for i in range(len(data['train']['x'][0])):
        header.append('[x]')
    data['train']['x'].insert(0,header)
    data['train']['y'].insert(0,'[y]')
    data['test']['x'].insert(0,header)
    data['test']['y'].insert(0,'[y]')


    writeToFile(filename+'_train.csv',data['train']['x'],data['train']['y'])
    writeToFile(filename+'_test.csv',data['test']['x'],data['test']['y'])


def separateToTestTrain(x, y, factor=0.6):
    l = []
    for i in range(len(x)):
        l.append((x[i],y[i]))

    numpy.random.shuffle(l)
    x=[]
    y=[]
    for i in l:
        x.append(i[0])
        y.append(i[1])


    v = int(len(x)*factor)
    x_test = x[:v]
    x_train = x[v:]
    y_test = y[:v]
    y_train = y[v:]

    return  {'test': {'x': x_test, 'y': y_test}, 'train': {'x': x_train, 'y': y_train}}

filename=sys.argv[1]

x = []
y = []

with open(filename, 'r') as f:
    for line in f:
        line = list(map(float, line.split(',')))
        x.append(line[:-1])
        y.append(line[-1])

data = separateToTestTrain(x,y)
exportData(filename,data)
