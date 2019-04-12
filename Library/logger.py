"""
The purpose of this module is to create a portable, simple, but scalable logging functionality
in order to get away from the print statements
"""
import os
parent = os.path.dirname(os.path.realpath(__file__))
logFolderPath = os.path.join(parent,'Logs')
import json
counter = 0
# Create the log folder if it doesn't already exist
if not os.path.exists(logFolderPath):
    os.mkdir(logFolder)

def deleteLogs():
    if os.path.exists(logFolderPath):
        for filename in os.listdir(logFolderPath):
            os.unlink(os.path.join(logFolderPath,filename))

def log(*args,pre='default'):
    """
    (log s *pre) This is the basic log, essentailly just a print statement that goes to
    the Logs directory
    log: str pre=str -> None
    Effect: A new file is created (./<pre>.log) that contains s
    """
    global counter
    filepath = os.path.join(logFolderPath, pre+'.log')       
    with open(filepath,'a+') as f:
        f.write(str(counter)+'. '+args.__str__()+'\n')
    counter += 1

def logJSON(*args,pre='default'):
    """
    (log s *pre) this logs an arbitrary object as a json with the provided prefix. If this file
    already exists, then it adds a number to the end
    logJSON str pre=str -> None
    Effect: A new file is created (./<pre><int>.json) that contains s
    """
    filepath = os.path.join(logFolderPath, pre+'.json')       
    if os.path.exists(filepath):
        n = 1
        filepath = filepath.replace('.json','1.json')
        while os.path.exists(filepath):
            filepath = filepath.replace(str(n)+'.json',str(n+1)+'.json')
            n += 1
    with open(filepath,'w+') as f:
        json.dump(args,f,indent=4)
