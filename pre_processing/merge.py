import csv
import os

# dir1 ="/home/nihali/work/5thsem/DataAnalytics/project"
# list_dir = os.listdir(dir1)
import glob
path = "/home/nihali/work/5thsem/DataAnalytics/project/*.csv"
with open('new/datasetmerge.csv','w') as csvv:
    csvwriter = csv.writer(csvv, delimiter =',', quoting=csv.QUOTE_MINIMAL)
    print("hello")
    for fname in glob.glob(path):
        print(fname)
        with open(fname,'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter =',', quoting=csv.QUOTE_MINIMAL )
            for row in csvreader:
                print(row)
                print(row[0])

                csvwriter.writerow(row)
