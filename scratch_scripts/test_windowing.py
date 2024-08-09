import csv

hist_data_file = 'data/BTC-2017min.csv'
decimals = 2
changes = []

with open(hist_data_file, mode='r') as file:
    csvFile = csv.reader(file)
    count = 0
    for lines in csvFile:
        if count < 100:
            print(lines)
            count += 1
        else:
            break

        # 5 hour change
        # diff5 = float(lines[29]) - float(lines[0])
        # percentagechange5 = (diff5 / float(lines[0]) * 25)
        # percentagechange5 = round(percentagechange5 * decimals) / decimals
        # percentagechange5 += 0.5
        # if (percentagechange5 < 0): percentagechange5 = 0
        # if (percentagechange5 > 1): percentagechange5 = 1
        # changes.append(percentagechange5)
