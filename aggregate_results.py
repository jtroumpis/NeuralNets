import json, math

def addToD(jason):
    d = {jason['c']: jason['error']}
    return d

nn_d = {}
with open('complete_res.json', 'r') as f:
    for line in f:
        datum = json.loads(line)
        if datum['nn'] not in nn_d:
            nn_d[datum['nn']] = {'test': {}, 'train': {} }
        if datum['testing']:
            try:
                nn_d[datum['nn']]['test'][datum['c']]['values'].append(datum['error'])
                nn_d[datum['nn']]['test'][datum['c']]['mean'] += datum['error']/10
            except KeyError:
                nn_d[datum['nn']]['test'][datum['c']] = {'values': [datum['error']], 'mean': datum['error']/10}
        else:
            try:
                nn_d[datum['nn']]['train'][datum['c']]['values'].append(datum['error'])
                nn_d[datum['nn']]['train'][datum['c']]['mean'] += datum['error']/10
            except KeyError:
                nn_d[datum['nn']]['train'][datum['c']] = {'values': [datum['error']], 'mean': datum['error']/10}
# print(nn_d)

for nn in nn_d:
    for test_type in nn_d[nn]:
        for c in nn_d[nn][test_type]:
            std = 0
            for value in nn_d[nn][test_type][c]['values']:
                std += (nn_d[nn][test_type][c]['mean'] - value)**2
            std = std / 9
            nn_d[nn][test_type][c]['std'] = math.sqrt(std)

for nn in nn_d:
    for test_type in nn_d[nn]:
        for key, value in nn_d[nn][test_type].items():
            print(nn,",",test_type,",",key,",",value['mean'],",",value['std'])
