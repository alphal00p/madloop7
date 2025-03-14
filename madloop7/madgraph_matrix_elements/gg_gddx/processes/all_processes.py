from __future__ import division
from model.aloha_methods import *
from model.wavefunctions import *
class Matrix_1_gg_gddx(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def get_external_masses(self, model):

        return ( (model.ZERO, model.ZERO), (model.ZERO, model.ZERO, model.ZERO) )

    def smatrix(self,p, model):
        #  
        #  MadGraph5_aMC@NLO v. 3.5.7, 2024-11-29
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: g g > g d d~ WEIGHTED<=3 @1
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = 5
        ndiags = 16
        ncomb = 32
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,1,1],
        [-1,-1,-1,1,-1],
        [-1,-1,1,-1,1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,1,1],
        [-1,-1,1,1,-1],
        [-1,1,-1,-1,1],
        [-1,1,-1,-1,-1],
        [-1,1,-1,1,1],
        [-1,1,-1,1,-1],
        [-1,1,1,-1,1],
        [-1,1,1,-1,-1],
        [-1,1,1,1,1],
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,-1],
        [1,-1,-1,1,1],
        [1,-1,-1,1,-1],
        [1,-1,1,-1,1],
        [1,-1,1,-1,-1],
        [1,-1,1,1,1],
        [1,-1,1,1,-1],
        [1,1,-1,-1,1],
        [1,1,-1,-1,-1],
        [1,1,-1,1,1],
        [1,1,-1,1,-1],
        [1,1,1,-1,1],
        [1,1,1,-1,-1],
        [1,1,1,1,1],
        [1,1,1,1,-1]]
        denominator = 256
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0.] * ndiags
        self.helEvals = []
        ans = 0.
        for hel in helicities:
            t = self.matrix(p, hel, model)
            ans = ans + t
            self.helEvals.append([hel, t.real / denominator ])
        ans = ans / denominator
        return ans.real

    def matrix(self, p, hel, model):
        #  
        #  MadGraph5_aMC@NLO v. 3.5.7, 2024-11-29
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: g g > g d d~ WEIGHTED<=3 @1
        #  
        #  
        # Process parameters
        #  
        ngraphs = 18
        nexternal = 5
        nwavefuncs = 12
        ncolor = 6
        ZERO = 0.
        #  
        # Color matrix
        #  
        denom = [9,9,9,9,9,9];
        cf = [[64,-8,-8,1,1,10],
        [-8,64,1,10,-8,1],
        [-8,1,64,-8,10,1],
        [1,10,-8,64,1,-8],
        [1,-8,10,1,64,-8],
        [10,1,1,-8,-8,64]];
        #
        # Model parameters
        #
        
        GC_12 = model.GC_12
        GC_11 = model.GC_11
        GC_10 = model.GC_10
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w[0] = vxxxxx(p[0],ZERO,hel[0],-1)
        w[1] = vxxxxx(p[1],ZERO,hel[1],-1)
        w[2] = vxxxxx(p[2],ZERO,hel[2],+1)
        w[3] = oxxxxx(p[3],ZERO,hel[3],+1)
        w[4] = ixxxxx(p[4],ZERO,hel[4],-1)
        w[5]= VVV1P0_1(w[0],w[1],GC_10,ZERO,ZERO)
        w[6]= FFV1_1(w[3],w[2],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[4],w[6],w[5],GC_11)
        w[7]= FFV1_2(w[4],w[2],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV1_0(w[7],w[3],w[5],GC_11)
        w[8]= FFV1P0_3(w[4],w[3],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 3
        amp[2]= VVV1_0(w[5],w[2],w[8],GC_10)
        w[5]= VVV1P0_1(w[0],w[2],GC_10,ZERO,ZERO)
        w[9]= FFV1_1(w[3],w[1],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 4
        amp[3]= FFV1_0(w[4],w[9],w[5],GC_11)
        w[10]= FFV1_2(w[4],w[1],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 5
        amp[4]= FFV1_0(w[10],w[3],w[5],GC_11)
        # Amplitude(s) for diagram number 6
        amp[5]= VVV1_0(w[5],w[1],w[8],GC_10)
        w[5]= FFV1_1(w[3],w[0],GC_11,ZERO,ZERO)
        w[11]= VVV1P0_1(w[1],w[2],GC_10,ZERO,ZERO)
        # Amplitude(s) for diagram number 7
        amp[6]= FFV1_0(w[4],w[5],w[11],GC_11)
        # Amplitude(s) for diagram number 8
        amp[7]= FFV1_0(w[10],w[5],w[2],GC_11)
        # Amplitude(s) for diagram number 9
        amp[8]= FFV1_0(w[7],w[5],w[1],GC_11)
        w[5]= FFV1_2(w[4],w[0],GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 10
        amp[9]= FFV1_0(w[5],w[3],w[11],GC_11)
        # Amplitude(s) for diagram number 11
        amp[10]= FFV1_0(w[5],w[9],w[2],GC_11)
        # Amplitude(s) for diagram number 12
        amp[11]= FFV1_0(w[5],w[6],w[1],GC_11)
        # Amplitude(s) for diagram number 13
        amp[12]= VVV1_0(w[0],w[11],w[8],GC_10)
        # Amplitude(s) for diagram number 14
        amp[13]= FFV1_0(w[7],w[9],w[0],GC_11)
        # Amplitude(s) for diagram number 15
        amp[14]= FFV1_0(w[10],w[6],w[0],GC_11)
        w[10]= VVVV1P0_1(w[0],w[1],w[2],GC_12,ZERO,ZERO)
        w[6]= VVVV3P0_1(w[0],w[1],w[2],GC_12,ZERO,ZERO)
        w[9]= VVVV4P0_1(w[0],w[1],w[2],GC_12,ZERO,ZERO)
        # Amplitude(s) for diagram number 16
        amp[15]= FFV1_0(w[4],w[3],w[10],GC_11)
        amp[16]= FFV1_0(w[4],w[3],w[6],GC_11)
        amp[17]= FFV1_0(w[4],w[3],w[9],GC_11)

        jamp = [None] * ncolor

        jamp[0] = +complex(0,1)*amp[1]+amp[2]+complex(0,1)*amp[6]-amp[8]+amp[12]+amp[15]-amp[17]
        jamp[1] = +complex(0,1)*amp[4]+amp[5]-complex(0,1)*amp[6]-amp[7]-amp[12]-amp[15]-amp[16]
        jamp[2] = -complex(0,1)*amp[1]-amp[2]+complex(0,1)*amp[3]-amp[5]-amp[13]+amp[16]+amp[17]
        jamp[3] = -complex(0,1)*amp[3]+amp[5]+complex(0,1)*amp[9]-amp[10]-amp[12]-amp[15]-amp[16]
        jamp[4] = +complex(0,1)*amp[0]-amp[2]-complex(0,1)*amp[4]-amp[5]-amp[14]+amp[16]+amp[17]
        jamp[5] = -complex(0,1)*amp[0]+amp[2]-complex(0,1)*amp[9]-amp[11]+amp[12]+amp[15]-amp[17]

        self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        self.amp2[1]+=abs(amp[1]*amp[1].conjugate())
        self.amp2[2]+=abs(amp[2]*amp[2].conjugate())
        self.amp2[3]+=abs(amp[3]*amp[3].conjugate())
        self.amp2[4]+=abs(amp[4]*amp[4].conjugate())
        self.amp2[5]+=abs(amp[5]*amp[5].conjugate())
        self.amp2[6]+=abs(amp[6]*amp[6].conjugate())
        self.amp2[7]+=abs(amp[7]*amp[7].conjugate())
        self.amp2[8]+=abs(amp[8]*amp[8].conjugate())
        self.amp2[9]+=abs(amp[9]*amp[9].conjugate())
        self.amp2[10]+=abs(amp[10]*amp[10].conjugate())
        self.amp2[11]+=abs(amp[11]*amp[11].conjugate())
        self.amp2[12]+=abs(amp[12]*amp[12].conjugate())
        self.amp2[13]+=abs(amp[13]*amp[13].conjugate())
        self.amp2[14]+=abs(amp[14]*amp[14].conjugate())
        matrix = 0.
        for i in range(ncolor):
            ztemp = 0
            for j in range(ncolor):
                ztemp = ztemp + cf[i][j]*jamp[j]
            matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        return matrix

