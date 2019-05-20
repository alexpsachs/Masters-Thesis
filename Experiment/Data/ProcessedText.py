"""
The purpose of this script is to process the text in ./CleanText to obtain the
big 5 and SYMLOG values
"""
import os,sys,json,code,time
HOME = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(HOME,'Library')
DATA = os.path.join(HOME,'Experiment','Data')
sys.path.append(LIB)
import IBM
import SYMLOG
import logger
def log(*args,pre=None):
    logger.log(*args,pre=('ProcessedText.py' if pre == None else 'ProcessedText.py.'+pre))

# 0. setup the paths
# INDIR = '/home/a2sachs/Documents/Experiment2.2/Data/CleanText'
INDIR = os.path.join(DATA,'CleanText')
# OUTDIR = '/home/a2sachs/Documents/Experiment2.2/Data/ProcessedText_Big5'
OUTDIR = os.path.join(DATA,'ProcessedText_Big5')
IBMDIR = os.path.join(DATA,'ProcessedText_IBM_raw')
# SYMDIR = '/home/a2sachs/Documents/Experiment2.2/Data/ProcessedText_Sym'
SYMDIR = os.path.join(DATA,'ProcessedText_Sym')

#FUNCTIONS
def processBig5(filenames):
    """
    This function processes the text in ./CleanText into ./ProcessedText_Big5/<reponame>.json
    Output of the form {name:{big5Attribute:rawScore(0-1)}}
    """
    for filename in filenames:
        indata = json.load(open(os.path.join(INDIR,filename),'r'))
        outpath = os.path.join(OUTDIR,filename)
        outdata = {}
        for name,text in indata.items():
            raw_json_path = os.path.join(IBMDIR,filename[:-5]+'.'+name+'.json')
            big5 = IBM.getBig5FromText(text,filename=raw_json_path)
            if big5 != None:
                outdata[name] = big5
        json.dump(outdata,open(outpath,'w'),indent=4)
        log('finished Big5 for',filename)

def processSYMLOG(filenames):
    """
    This function processes the text in ./CleanText into ./ProcessedText/<reponame>.json (adds to the file)
    Output of the form {name:{big5Attribute:rawScore(0-1)}}
    """
    pre = 'processSYMLOG'
    log('start',pre=pre)
    for filename in filenames:
        log('starting file',filename,pre=pre)
        indata = json.load(open(os.path.join(OUTDIR,filename),'r'))
        outdata = {}
        if indata != {}: # sometimes the big5 json is empty, in that case we make sym to be empty too
            big5Attributes = [
                'openness',
                'conscientiousness',
                'extraversion',
                'agreeableness',
                'neuroticism'
                ]
            big5 = {name:{key:val for key,val in data.items() if key in big5Attributes} for name,data in indata.items()}
            outpath = os.path.join(SYMDIR,filename)
            sym = SYMLOG.convertBig5ToSYMLOG(big5) # basic SYMLOG attributes
            log('sym.values',sym.values(),pre=pre)
            symAttributes = [x for x in list(sym.values())[0]]
            for name in indata:
                outdata[name] = {}
                for attribute in symAttributes:
                    outdata[name][attribute] = sym[name][attribute]
        else:
            log(filename,'did not have any big5 data',pre=pre)
        json.dump(outdata,open(outpath,'w'),indent=4)
        log('finished SYMLOG for',filename,pre=pre)

if __name__ == '__main__':
    print('made it')
    logger.deleteLogs()
    start = time.time()
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    if not os.path.exists(SYMDIR):
        os.mkdir(SYMDIR)
    if not os.path.exists(IBMDIR):
        os.mkdir(IBMDIR)
    # 1. Setup the filepaths
    all_filenames = os.listdir(INDIR)
    print('all_filenames',len(all_filenames))
    # 1.a. exclude repos already done
    filenames = all_filenames
    print('filenames',len(filenames))

    # 2. process the text
    processBig5(filenames)
    processSYMLOG(filenames)
    end = time.time()
    print('completed in',end-start,'seconds')
