"""
This script is intended to strip the aggregated text from 
'Experiment/Data/AggregateText'
of all code and place the content into
'Experiment/Data/CleanText'
"""
import os,sys,json,time
HOME = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(HOME,'Library')
sys.path.append(LIB)
import markdown_to_text
import logger
def log(*args):
    logger.log(*args,pre='CleanText.py')
logger.deleteLogs()

# 0. create the paths
INDIR = os.path.join(HOME,'Experiment','Data','AggregateText')
# OUTDIR = '/home/a2sachs/Documents/Experiment2.2/Data/CleanText'
OUTDIR = os.path.join(HOME,'Experiment','Data','CleanText')
print('path exists',os.path.exists(OUTDIR))
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)
    print('made directory',OUTDIR)

# 1. commence cleaning for each and eveyr file
start = time.time()
for filename in os.listdir(INDIR):
    inpath = os.path.join(INDIR,filename)
    outpath = os.path.join(OUTDIR,filename)
    indata = json.load(open(inpath,'r'))
    outdata = {}
    for person,text in indata.items():
        clean_text = markdown_to_text.markdown_to_text(text)
        outdata[person] = clean_text
    json.dump(outdata,open(outpath,'w'),indent=4)
    print('cleaned',inpath,'into',outpath)
    log('finished cleaning',inpath)
end = time.time()
print('all done')
print('cleaning took',end-start,'seconds')
