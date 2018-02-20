import csv
def readCSV(filename = 'data.csv', keep_this=None):
    d = {}
    with open(filename) as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f)
        # headers = reader.next()
        # print(headers)
        # for row in reader:
        #     print (row)

        for row in reader:
            # print (row['time'])
            d[row['time']] = {}

            for key, value in row.items():
                try:
                    value = int(value)
                except ValueError:
                    pass
                except TypeError:
                    break

                if keep_this:
                    if key in keep_this:
                        d[row['time']][key]=(value)
                else:
                    d[row['time']][key]=(value)
        return d

print(readCSV(keep_this=['CPU','MemoryUsed']))
