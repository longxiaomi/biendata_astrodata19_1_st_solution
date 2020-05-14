import numpy as np
import random

def getdata(filename,sheets,shuffle=True):
    rows=[]
    for i in sheets:
        data = np.memmap(filename, dtype='float32', mode='r',shape=(10000, 2601),offset=i * 10000 * 2601 * 4)
        rows=rows+list(data)
    if shuffle:
        random.shuffle(rows)
    return [np.array(i)[:2600] for i in rows],[[int(np.array(i)[2600])] for i in rows]
def gettestdata(filename,sheets):
    rows=[]
    for i in sheets:
        data = np.memmap(filename, dtype='float32', mode='r',shape=(10000, 2600),offset=i * 10000 * 2600 * 4)
        rows=rows+list(data)
    return [np.array(i) for i in rows if i[0]!=0]
def gettraindata(filename):
    inputs=[]
    lables=[]
    with open(filename) as f:
        while True:
            line=f.readline().strip('\n')
            temp=line.split(',')
            if line=='':
                break
            if temp[-2] in ['qso','star','galaxy']:
                inputs.append([float(i) for i in temp[0:2600]])
                if temp[-2]=='qso':
                    lables.append([2])
                if temp[-2]=='star':
                    lables.append([0])
                if temp[-2]=='galaxy':
                    lables.append([1])
    return inputs,lables
    
def getcsvdata(filename):
    ret=[]
    id=[]
    with open(filename) as f:
        while True:
            temp = f.readline()
            if temp:
                row=temp.strip('\n').split(',')
                if row[0]!='FE0' and len(row)>2600:
                    ret.append([float(i) for i in row[:2600]])
                    id.append(row[2600])
            else:
                break
    return ret,id
def getlables(filename):
    labels=[]
    with open(filename) as f:
        lines=f.readlines()
    for i in range(1,len(lines)):
        label=lines[i].strip('\n').split(',')[1]
        if label=='star':
            labels.append([0])
        if label=='galaxy':
            labels.append([1])
        if label=='qso':
            labels.append([2])
    return labels

def getnumpydata(filename):
    data = np.load(filename)
    print(data.shape)
    return inputs,lables