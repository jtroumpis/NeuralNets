import sys, numpy

def writeToFile(filename,x,y):
    with open(filename,'w') as f:
        for i in range(len(x)):
            for j in x[i]:
                f.write(str(j)+', ')
            f.write(str(y[i])+'\n')

def exportData(filename,data):
    filename = filename.split('.')[0]
    filename = filename.split('/')
    filename.insert(1,'ready')
    filename = '/'.join(filename)
    # print('/'.join(temp))


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
    x_train = x[:v]
    x_test = x[v:]
    y_train = y[:v]
    y_test = y[v:]

    return  {'test': {'x': x_test, 'y': y_test}, 'train': {'x': x_train, 'y': y_train}}

def readDat(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            line = list(map(float, line.split(',')))
            x.append(line[:-1])
            y.append(line[-1])
    return x,y

def readCSV(filename):
    import csv
    x = []
    y = []
    find_next = False
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', dialect='excel')
        prev_row = None
        headers = spamreader.next()
        for row in spamreader:
            try:
                if row[-1] is '': 
                    find_next = False
                    continue

                for i in range(len(row)):
                    row[i] = row[i].replace(',','.')
                    if row[i] is '':
                        row[i] = prev_row[i]
                    row[i] = abs(float(row[i]))

                print(row)
                x.append(row[:-1])
                y.append(row[-1])
                prev_row = row
            except ValueError:
                find_next = True
    return x,y

filename=sys.argv[1]
x,y = readCSV(filename)
data = separateToTestTrain(x,y)
exportData(filename,data)
