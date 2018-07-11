import sys

filename=sys.argv[1]

x = []
y = []

with open(filename, 'r') as f:
    for line in f:
        line = list(map(float, line.split(',')))
        x.append(line[:-1])
        y.append(line[-1])
