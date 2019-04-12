"""
This module is respnsible for all of the SYMLOG calculations.
"""
#Imports
import math
import json
import os
import pandas as pd
import statsmodels.api as sm
import sklearn
from sklearn import linear_model
import code
import matplotlib.pyplot as plt
import logger
def log(*args,pre=None):
    logger.log(*args,pre=('SYMLOG.py' if pre==None else 'SYMLOG.py.'+pre))
LIB = '/home/a2sachs/Documents/Library'

#Constants
file_path = os.path.abspath(__file__)
parent = os.path.dirname(file_path)
SYMLOG_LABELS = json.load(open(os.path.join(parent,'SYMLOG_LABELS.json'),'r'))
SYMLOG_POLES = json.load(open(os.path.join(parent,'SYMLOG_POLES.json'),'r'))
SYMLOG_AXES = json.load(open(os.path.join(parent,'SYMLOG_AXES.json'),'r'))
SYMLOG_SCG_OPT = json.load(open(os.path.join(parent,'SYMLOG_SCG_OPT.json'),'r')) #SCG optimum profile
SYMLOG_SCG_OPT = {key:val/33 for key,val in SYMLOG_SCG_OPT.items()} # normalize from 0 to 1 from original 0 to 33
def calc(s,d):
    first = s[0]
    second = s[2]
    def sumKey(c,d):
        return sum([val for key,val in d.items() if c in key.lower()]) # ranges from 0 to 9
    initial = sumKey(first,d) - sumKey(second,d) # ranges from -9 to 9 so double to get the compatible range of -18 to 18
    ans = initial * 2
    return ans
SYMLOG_SCG_OPT_posn = {
    'p_n':calc('p_n',SYMLOG_SCG_OPT),
    'f_b':calc('f_b',SYMLOG_SCG_OPT),
    'u_d':calc('u_d',SYMLOG_SCG_OPT)
    }

#Debug function
Debug = False
def p(*args):
    if Debug:
        print(*args)

#Classes
class SYMLOGPlot:
    """
    This class contains the plot, personalities, compass, and analysis methods for
    a single repository
    """
    def __init__(self,personalities,compassMethod='regression'):
        """
        peronalities = {person;{SYMLOGAxis;score_from_-18_to_+18, SYMLOGPoles:score_from_0_to_18}}

        compassMethod: (anyof 'regression' 'PF')

        If compassMethod is 'regression' then a linear regression will be done on the personalities
        to determine an orientation of best fit

        If compassMethod is 'PF' then the compass will be oriented according to Bales' ideal which
        places the reference circle in the middle of the PF region
        """
        pre = 'SYMLOGPlot.__init__'
        log('Initialized with',personalities,compassMethod,pre=pre)
        self.angle = math.pi/4 # Bale's ideal for compasss placememt
        d = {
                'Person':[person for person in personalities],
                'p':[attributes['p'] for person,attributes in personalities.items()],
                'n':[attributes['n'] for person,attributes in personalities.items()],
                'f':[attributes['f'] for person,attributes in personalities.items()],
                'b':[attributes['b'] for person,attributes in personalities.items()],
                'u':[attributes['u'] for person,attributes in personalities.items()],
                'd':[attributes['d'] for person,attributes in personalities.items()],
                'p_n':[attributes['p_n'] for person,attributes in personalities.items()],
                'f_b':[attributes['f_b'] for person,attributes in personalities.items()],
                'u_d':[attributes['u_d'] for person,attributes in personalities.items()]
            }
        self.personalities = pd.DataFrame(data=d)
        self.personalitiesDict = personalities
        if compassMethod == 'regression':
            X = self.personalities['p_n']
            Y = self.personalities['f_b']
            # X = sm.add_constant(X) # exclude the constant because this insures that the y-intercept is at 0
            log('X',X,pre=pre)
            log('Y',Y,pre=pre)
            model = sm.OLS(Y,X).fit()
            # print(model.summary())
            coeff = model.params.values[0]
            orig_angle = math.atan(coeff)
            log('coeff',coeff,pre=pre)
            log('orig_angle',orig_angle/math.pi,pre=pre)
            # self.angle = (math.atan(1/coeff) if coeff > 0 else math.pi-math.atan(-1/coeff))
            self.angle = orig_angle if coeff > 0 else math.pi+orig_angle

            # predictions = model.predict(X)
            # path = '/home/a2sachs/test.json'
            # json.dump([list(X.values),list(Y.values),list(predictions)],open(path,'w'),indent=4)

        radius = 9
        x = math.cos(self.angle) * radius
        y = math.sin(self.angle) * radius
        self.ref = (x,y)
        self.opp = (-x,-y)
        # record the calculated variables
        log('angle',self.angle/math.pi,pre=pre)
        log('ref',self.ref,pre=pre)
        log('opp',self.opp,pre=pre)
        
    def draw(self,filename=None):
        """
        (draw) -> None: This function is intended to draw out what the current state of the plot is
        """
        size_coefficient = 15
        # plt.plot(self.personalities['p_n'],self.personalities['f_b'],'ro',
        #         s=[(x+18)*5 for x in self.personalities['u_d'].values])
        names = list(self.personalities['Person'].values)
        x = list(self.personalities['p_n'].values)
        y = list(self.personalities['f_b'].values)
        z = [(x+18)*size_coefficient for x in self.personalities['u_d'].values]
        plt.scatter(x,y,s=z)
        # label the points
        for i,name in enumerate(names):
            plt.annotate(s=name,xy=(x[i],y[i]))
        plt.xlabel('p_n')
        plt.ylabel('f_b')
        plt.axis('scaled')
        plt.ylim((-18,18))
        plt.xlim((-18,18))
        # draw axis lines
        plt.plot([-18,18],[0,0],'k-')
        plt.plot([0,0],[-18,18],'k-')
        # draw the line of polarization
        lineLength = (18*2)
        start = (math.cos(self.angle)*lineLength/2, math.sin(self.angle)*lineLength/2)
        end = (-start[0],-start[1])
        plt.annotate(s='', xy=start, xytext=end, arrowprops=dict(arrowstyle='<->'))
        # draw the reference and opposition circles
        refCircle = plt.Circle(self.ref,radius=9,edgecolor='black',fill=False)
        plt.gca().add_patch(refCircle)
        refInnerCircle = plt.Circle(self.ref,radius=4.5,edgecolor='black',fill=False)
        plt.gca().add_patch(refInnerCircle)
        oppCircle = plt.Circle(self.opp,radius=9,edgecolor='black',fill=False)
        plt.gca().add_patch(oppCircle)
        oppInnerCircle = plt.Circle(self.opp,radius=4.5,edgecolor='black',fill=False)
        plt.gca().add_patch(oppInnerCircle)
        # label the circles
        plt.annotate('Inner Circle',self.ref)
        # plt.gca().add_patch(polLine)
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename)
        # code.interact(local=dict(globals(),**locals()))
        plt.close()

    def getMembersByRegion(self,region):
        """
        (getMembersByRegion region) -> {person;{SYMLOGAxis;score_from_-18_to_+18, SYMLOGPoles:score_from_0_to_18}}

        This function obtains all of the members that are contained within the region specified. The regions are:
            * 'ref': the reference circle
            * 'inner': the inner core group (the small circle enclosed by the reference circle
            * 'opp': the opposition circle
            * 'opp core': the inner core of the opposition circle
        """
        pre = 'getMembersByRegion'
        log('start',pre=pre)
        def dist(posn0,posn1):
            return math.sqrt((posn1[0]-posn0[0])**2 + (posn1[1]-posn0[1])**2)
        if region == 'inner':
            center = self.ref
            ans = {key:val for key,val in self.personalitiesDict.items() 
                    if dist((val['p_n'],val['f_b']),center) <= 4.5}
            return ans
        elif region == 'ref':
            center = self.ref
            ans = {key:val for key,val in self.personalitiesDict.items() 
                    if 4.5 < dist((val['p_n'],val['f_b']),center) <= 9}
            return ans
        elif region == 'opp':
            center = self.opp
            ans = {key:val for key,val in self.personalitiesDict.items() 
                    if 4.5 < dist((val['p_n'],val['f_b']),center) <= 9}
            return ans
        elif region == 'opp core':
            center = self.opp
            ans = {key:val for key,val in self.personalitiesDict.items() 
                    if dist((val['p_n'],val['f_b']),center) <= 4.5}
            return ans
        else:
            log('error unknown region',region,pre=pre)
            return None

    def getSCGDeviation(self):
        """
        (getSCGDeviation) -> float

        This function is intended to calculate the variance of this group's members from the SYMLOG_SCG_OPT 
        (the SCG optimum bar graph for an ideal member) and then return the total mean squared error 
        (or None if there is only one person in the group) where
        error is considered the deviation from this optimum
        """
        pre = 'getSCGDeviation'
        log('start',pre=pre)
        dimensions = ['p_n','f_b','u_d']
        posn1 = [SYMLOG_SCG_OPT_posn[dimension] for dimension in dimensions]
        def calcDist(posn0,posn1):
            squares = sum([(posn1[i]-posn0[i])**2 for i,_ in enumerate(posn0)])
            return math.sqrt(squares)
        sumOfSquares = sum([calcDist(posn0,posn1) for posn0 in self.personalities[dimensions].values])
        log('sumOfSquares',sumOfSquares,pre=pre)
        n = self.personalities.shape[0]
        log('n',n,pre=pre)
        if n == 1:
            return None
        variance = sumOfSquares/(n-1)
        log('variance',variance,pre=pre)
        # code.interact(local=dict(globals(),**locals()))
        return variance

    def getUPFCorrelation(self,filename_prefix=None):
        """
        (getUPFCorrelation) -> Float

        This function calculates the correlation between U,P, and F dimensions and returns the average of the
        pearson r correlations between UP, UF, and PF

        kargs:
            filename_prefix: str    If this is provided then plots for each correlation will be saved to this prefix
        """
        U = self.personalities['u_d']
        P = self.personalities['p_n']
        F = self.personalities['f_b']

        def getCorr(X,Y,filename=None):
            x = [[e] for e in X.values]
            y = [e for e in Y.values]
            regr = sklearn.linear_model.LinearRegression()
            regr.fit(x,y)
            predY = regr.predict(x)
            corr = sklearn.metrics.r2_score(y,predY)
            if filename != None:
                # plot the correlations
                plt.plot([e for e in X.values],[e for e in Y.values],'ko')
                plt.plot([e for e in X.values],[e for e in predY])
                plt.savefig(filename)
                plt.close()
            return float(corr)

        comparisons_data = [(U,P),(U,F),(P,F)]
        comparisons_strs = [('U','P'),('U','F'),('P','F')]
        comparisons_corrs = []
        for inputs,strs in zip(comparisons_data,comparisons_strs):
            filename0 = None if filename_prefix == None else filename_prefix+'_'+'_'.join(strs)+'.png'
            filename1 = None if filename_prefix == None else filename_prefix+'_'+strs[1]+'_'+strs[0]+'.png'
            corr0 = getCorr(*inputs,filename=filename0)
            corr1 = getCorr(*inputs[::-1],filename=filename1)
            tolerance = 1*10**-10
            if abs(corr0 - corr1) > tolerance:
                print('ERROR correlations are not the same!!!!')
                code.interact(local=dict(globals(),**locals()))
            else:
                comparisons_corrs.append(corr0)

        avg = sum(comparisons_corrs)/3
        avg1 = sum([getCorr(x,y) for x,y in [(U,P),(U,F),(P,F)]])/3
        avg2 = sum([getCorr(x,y) for x,y in [(P,U),(F,U),(F,P)]])/3
        # code.interact(local=dict(globals(),**locals()))
        return avg

    def reorientCompass(self):
        """
        (reorientCompass) This function checks which interpretation of the compass provides the greater
        number of participants in the ref circle and then reorients the compass to this interpretation
        """
        num_in_ref = len(list(self.getMembersByRegion('ref').keys()))
        num_in_opp = len(list(self.getMembersByRegion('opp').keys()))
        if num_in_opp > num_in_ref:
            self.ref,self.opp = self.opp,self.ref
            self.angle += math.pi

#Functions
def convertBig5ToSYMLOG(big5Personas):
    """{person;{big5Attribute;score_from_0_to_1}} -> {person;{SYMLOGAxis;score_from_-18_to_+18, SYMLOGPoles:score_from_0_to_18}}

    This function is responsible for utilizing SYMLOG consulting group's mapping to convert
    a dict of big5 score into SYMLOG scores which are:

    p: Positive/Friendly
    n: Negative/Unfriendly
    f: Forward/Accepting of authority
    b: Backward/Rejects authority
    u: Up/Dominance
    d: Down/Submissiveness
    p_n: The final location on the positive-negative plane
    f_b: The final location on the forward-backward plane
    u_d: The final location on the up-down plane
    """
    ans = {}
    for person,persona in big5Personas.items():
        if persona == 'N/A':
            ans[person] = 'N/A'
        else:
            def convert(att1,att2):
                val1 = persona[att1]
                val2 = persona[att2]
                # return (val1*18)/2 + (val2*18)/2
                constant = math.cos(math.pi/4)
                return constant*val1*18*0.5 + constant*val2*18*0.5
            p = convert('agreeableness','openness')
            n = convert('conscientiousness','neuroticism')
            f = convert('conscientiousness','agreeableness')
            b = convert('neuroticism','openness')
            ext = persona['extraversion']
            u = 0 if 0 <= ext < 0.5 else ext-0.5*2*18
            d = ext*2*18 if 0 <= ext < 0.5 else 0

            p_n = p-n
            f_b = f-b
            u_d = u-d

            ans[person] = {
                'p_n':p_n,'f_b':f_b,'u_d':u_d,
                'p':p,'n':n,'f':f,'b':b,'u':u,'d':d
            }
    return ans

def getSYMLOGVectors(rawVectorCounts,conversationCounts):
    """((dictof str:(dictof str:int)),(dictof str:int)) -> (dictof str:(dictof str:number))

    This function is responsible for converting data on people into their SYMLOG vectors

    This function takes in a dict (rawVectorCounts) of the form {person_name:{SYMLOG_label:raw_count_of_label}}
    and a dict (conversationCounts) of the form {person_name:number_of_conversations} and returns a
    dict of the form {person_name:{(anyof 'u_d','f_b','p_n'):that_pole's_score}}.
    """
    ans = {}
    data = rawVectorCounts
    for person,entry in rawVectorCounts.items():
        ans[person] = getSYMLOGVector(entry,conversationCounts[person])
    return ans

def getSYMLOGVector(vectorCounts,conversationCount):
    """(dictof str:int) -> (dictof str:number)

    This function is responsible for converting data on a person into their SYMLOG vectors

    This function takes in a dict (vectorCounts) of the form {SYMLOG_label:raw_count_of_label}
    and a int (conversationCount) representing the conversations that was annotated
    for them and returns a dict of the form {(anyof 'u_d','f_b','p_n'):that_pole's_score}.
    """
    ans = {}
    data = vectorCounts
    #2 & 3. Normalize by conversation counts and max score
    for label in SYMLOG_LABELS:
        newLabel = '{0}_norm'.format(label)
        data[newLabel] = data[label]*2/conversationCount
    #4. Sum across each of the 6 poles
    for label in SYMLOG_POLES:
        #if the entry is a normalized one and is applicable to the current pole, then
        #include its value in the summation
        poleValue = sum([value for key, value in data.items()\
                if key.endswith('_norm') and key[:key.find('_')].find(label) != -1])
        data[label.upper()] = poleValue
    #5. Calculate the differences between opposing poles
    axes = SYMLOG_AXES
    for axis in axes:
        positiveDimension = axis[0].upper()
        negativeDimension = axis[2].upper()
        positiveValue = data[positiveDimension]
        negativeValue = data[negativeDimension]
        data[axis] = positiveValue - negativeValue
    #6 Aggregate the answer and return it
    ans = {key:val for key,val in data.items() if key in SYMLOG_AXES}
    p('ans',ans)
    return ans

# TESTING
if __name__ == '__main__':
    big5Persona1 = {
            'Bob':{
                'openness': 0.1,
                'conscientiousness': 0.2,
                'extraversion': 0.3,
                'agreeableness': 0.4,
                'neuroticism': 0.5
                },
            'Bill':{
                'openness': 0.5,
                'conscientiousness': 0.4,
                'extraversion': 0.3,
                'agreeableness': 0.2,
                'neuroticism': 0.1
                },
            'Nancy':{
                'openness': 0.2,
                'conscientiousness': 0.4,
                'extraversion': 0.8,
                'agreeableness': 1,
                'neuroticism': 0.9
                }
            }
    symPersona1 = {
        'Fred': {
            'p_n': 1, 
            'f_b': 2,
            'u_d': 0,
            'p': 3,
            'n': 2,
            'f': 2,
            'b': 0,
            'u': 5,
            'd': 5
            },
        'Tom': {
            'p_n': 1,
            'f_b': 1,
            'u_d': 3,
            'p': 1,
            'n': 0,
            'f': 2,
            'b': 1,
            'u': 6,
            'd': 3
            },
        'Febbie': {
            'p_n': -2,
            'f_b': -1, 
            'u_d': -3,
            'p': 2, 
            'n': 4,
            'f': 2, 
            'b': 3, 
            'u': 2, 
            'd': 5
            }
        }
    symPersona2 = {
        'Fred': {
            'p_n': 9, 
            'f_b': 9,
            'u_d': 0,
            'p': 3,
            'n': 2,
            'f': 2,
            'b': 0,
            'u': 5,
            'd': 5
            },
        'Tom': {
            'p_n': 7,
            'f_b': 9,
            'u_d': 3,
            'p': 1,
            'n': 0,
            'f': 2,
            'b': 1,
            'u': 6,
            'd': 3
            },
        'Febbie': {
            'p_n': -2,
            'f_b': -1, 
            'u_d': -3,
            'p': 2, 
            'n': 4,
            'f': 2, 
            'b': 3, 
            'u': 2, 
            'd': 5
            },
        'Veronica': {
            'p_n': -8,
            'f_b': -7, 
            'u_d': -9,
            'p': 2, 
            'n': 4,
            'f': 2, 
            'b': 3, 
            'u': 2, 
            'd': 5
            },
        'James': {
            'p_n': 10,
            'f_b': 12, 
            'u_d': -9,
            'p': 2, 
            'n': 4,
            'f': 2, 
            'b': 3, 
            'u': 2, 
            'd': 5
            }
        }
    expectedResult1 = {
        'Bob': {
            'p_n': -1.2727922061357853, 
            'f_b': 0.0,
            'u_d': -10.799999999999999,
            'p': 3.1819805153394642,
            'n': 4.4547727214752495,
            'f': 3.818376618407357,
            'b': 3.818376618407357,
            'u': 0,
            'd': 10.799999999999999
            },
        'Bill': {
            'p_n': 1.2727922061357853,
            'f_b': 0.0,
            'u_d': -10.799999999999999,
            'p': 4.4547727214752495,
            'n': 3.1819805153394642,
            'f': 3.818376618407357,
            'b': 3.818376618407357,
            'u': 0,
            'd': 10.799999999999999
            },
        'Nancy': {
            'p_n': -0.6363961030678933,
            'f_b': 1.9091883092036772, 
            'u_d': -17.2,
            'p': 7.636753236814714, 
            'n': 8.273149339882607,
            'f': 8.909545442950499, 
            'b': 7.000357133746822, 
            'u': -17.2, 
            'd': 0
            }
        }
    print('testing SYMLOG_SCG_OPT_posn')
    d = SYMLOG_SCG_OPT
    lst = [val for key,val in d.items() if 'p' in key.lower()]
    print('Passed' if len(lst) == 9 else 'Failed','len == ',len(lst))
    firstSum = sum([val for key,val in d.items() if 'p' in key.lower()])
    secondSum = sum([val for key,val in d.items() if 'n' in key.lower()])
    diff = (firstSum - secondSum) * 2
    print('Passed' if diff == SYMLOG_SCG_OPT_posn['p_n'] else 'Failed')
    print()

    print('testing convertBig5ToSYMLOG')
    ans = convertBig5ToSYMLOG(big5Persona1)
    print('Passed' if ans == expectedResult1 else 'Failed 1')
    print()
    
    print('testing SYMLOGPlot')
    sym = SYMLOGPlot(expectedResult1)
    sym.draw(filename=LIB+'/test_SYMLOG_expectedResult1.png')
    sym1 = SYMLOGPlot(symPersona1)
    sym1.draw(filename=LIB+'/test_SYMLOG_symPersona1.png')

    print('testing SYMLOGPlot.getSCGDeviation')
    sym1.getSCGDeviation()
    scg = SYMLOG_SCG_OPT_posn
    others = ['p','n','f','b','u','d']
    testPersona = {
            'a':{**scg,**{x:1 for x in others}},
            'b':{**scg,**{x:1 for x in others}},
            'c':{**scg,**{x:1 for x in others}}
            }
    sym = SYMLOGPlot(testPersona)
    ans = sym.getSCGDeviation()
    print('Passed 2' if ans == 0 else 'Failed 2')
    testPersona = {
            'a':{**{key:val+2 for key,val in scg.items()},**{x:1 for x in others}},
            'b':{**{key:val+2 for key,val in scg.items()},**{x:1 for x in others}},
            'c':{**{key:val+2 for key,val in scg.items()},**{x:1 for x in others}}
            }
    sym = SYMLOGPlot(testPersona)
    ans = sym.getSCGDeviation()
    exp = math.sqrt(4*3)*3/2
    print('Passed 3' if ans == exp else 'Failed 3 got {0} instead of {1}'.format(ans,exp))

    print('\ntesting SYMLOGPlot.getMembersByRegion')
    sym = SYMLOGPlot(expectedResult1)
    sym.draw(filename=LIB+'/test_SYMLOG_4.png')
    ans = sym.getMembersByRegion('ref')
    expKeys = ['Bob','Nancy']
    print('Passed 4' if list(ans.keys()) == expKeys else 'Failed 4')
    sym = SYMLOGPlot(symPersona2)
    sym.draw(filename=LIB+'/test_SYMLOG_symPersona2.png')
    ans = sym.getMembersByRegion('inner')
    expKeys = ['Fred','Tom']
    print('Passed 5' if list(ans.keys()) == expKeys else 'Failed 5')
    ans = sym.getMembersByRegion('ref')
    expKeys = ['James']
    print('Passed 6' if list(ans.keys()) == expKeys else 'Failed 6')
    ans = sym.getMembersByRegion('opp')
    expKeys = ['Febbie']
    print('Passed 7' if list(ans.keys()) == expKeys else 'Failed 7')
    ans = sym.getMembersByRegion('opp core')
    expKeys = ['Veronica']
    print('Passed 8' if list(ans.keys()) == expKeys else 'Failed 8')

    print('\ntesting SYMLOGPlot.getUPFCorrelation')
    sym = SYMLOGPlot(symPersona2)
    ans = sym.getUPFCorrelation(filename_prefix=LIB+'/test_SYMLOG_corr')
    exp = 0.4278666689281236
    print('Passed 9' if ans == exp else 'Failed 9')

    print('\nDone testing')
    code.interact(local=dict(globals(),**locals()))


