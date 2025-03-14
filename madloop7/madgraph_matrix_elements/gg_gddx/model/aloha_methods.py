from __future__ import division
from . import wavefunctions
import cmath

def FFV1_0(F1,F2,V3,COUP):
    TMP0 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-1j*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+1j*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+1j*(V3[4]))+F2[3]*(V3[2]+V3[5])))))
    vertex = COUP*-1j * TMP0
    return vertex



def FFV1_1(F2,V3,COUP,M1,W1):
    F1 = wavefunctions.WaveFunction(size=6)
    F1[0] = +F2[0]+V3[0]
    F1[1] = +F2[1]+V3[1]
    P1 = [-complex(F1[0]).real, -complex(F1[1]).real, -complex(F1[1]).imag, -complex(F1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    F1[2]= denom*1j*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-1j*(V3[4]))+(P1[2]*(+1j*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+1j*(V3[4]))+(P1[1]*(-1)*(V3[2]+V3[5])+(P1[2]*(-1)*(+1j*(V3[2]+V3[5]))+P1[3]*(V3[3]+1j*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))))
    F1[3]= denom*(-1j)*(F2[2]*(P1[0]*(-V3[3]+1j*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-1j*(V3[2])+1j*(V3[5]))+P1[3]*(V3[3]-1j*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1)*(V3[3]+1j*(V3[4]))+(P1[2]*(+1j*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+1j*(V3[4]))+F2[5]*(-V3[2]+V3[5]))))
    F1[4]= denom*(-1j)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+1j*(V3[4]))+(P1[2]*(-1)*(+1j*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+1j*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-1j*(V3[2])+1j*(V3[5]))-P1[3]*(V3[3]+1j*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+1j*(V3[4])))))
    F1[5]= denom*1j*(F2[4]*(P1[0]*(-V3[3]+1j*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1)*(+1j*(V3[2]+V3[5]))+P1[3]*(-V3[3]+1j*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+1j*(V3[4]))+(P1[2]*(-1j*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+1j*(V3[4]))+F2[3]*(V3[2]+V3[5]))))
    return F1



def FFV1_2(F1,V3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = +F1[0]+V3[0]
    F2[1] = +F1[1]+V3[1]
    P2 = [-complex(F2[0]).real, -complex(F2[1]).real, -complex(F2[1]).imag, -complex(F2[0]).imag]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*1j*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1)*(V3[3]+1j*(V3[4]))+(P2[2]*(+1j*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+(F1[3]*(P2[0]*(V3[3]-1j*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(+1j*(V3[2])-1j*(V3[5]))+P2[3]*(-V3[3]+1j*(V3[4])))))+M2*(F1[4]*(V3[2]-V3[5])+F1[5]*(-V3[3]+1j*(V3[4])))))
    F2[3]= denom*(-1j)*(F1[2]*(P2[0]*(-1)*(V3[3]+1j*(V3[4]))+(P2[1]*(V3[2]+V3[5])+(P2[2]*(+1j*(V3[2]+V3[5]))-P2[3]*(V3[3]+1j*(V3[4])))))+(F1[3]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]-1j*(V3[4]))+(P2[2]*(+1j*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+M2*(F1[4]*(V3[3]+1j*(V3[4]))-F1[5]*(V3[2]+V3[5]))))
    F2[4]= denom*(-1j)*(F1[4]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]+1j*(V3[4]))+(P2[2]*(-1j*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]-1j*(V3[4]))+(P2[1]*(-1)*(V3[2]+V3[5])+(P2[2]*(+1j*(V3[2]+V3[5]))+P2[3]*(V3[3]-1j*(V3[4])))))+M2*(F1[2]*(-1)*(V3[2]+V3[5])+F1[3]*(-V3[3]+1j*(V3[4])))))
    F2[5]= denom*1j*(F1[4]*(P2[0]*(-1)*(V3[3]+1j*(V3[4]))+(P2[1]*(V3[2]-V3[5])+(P2[2]*(+1j*(V3[2])-1j*(V3[5]))+P2[3]*(V3[3]+1j*(V3[4])))))+(F1[5]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-V3[3]+1j*(V3[4]))+(P2[2]*(-1)*(+1j*(V3[3])+V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[2]*(V3[3]+1j*(V3[4]))+F1[3]*(V3[2]-V3[5]))))
    return F2



def FFV1P0_3(F1,F2,COUP,M3,W3):
    V3 = wavefunctions.WaveFunction(size=6)
    V3[0] = +F1[0]+F2[0]
    V3[1] = +F1[1]+F2[1]
    P3 = [-complex(V3[0]).real, -complex(V3[1]).real, -complex(V3[1]).imag, -complex(V3[0]).imag]
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -1j* W3))
    V3[2]= denom*(-1j)*(F1[2]*F2[4]+F1[3]*F2[5]+F1[4]*F2[2]+F1[5]*F2[3])
    V3[3]= denom*(-1j)*(-F1[2]*F2[5]-F1[3]*F2[4]+F1[4]*F2[3]+F1[5]*F2[2])
    V3[4]= denom*(-1j)*(-1j*(F1[2]*F2[5]+F1[5]*F2[2])+1j*(F1[3]*F2[4]+F1[4]*F2[3]))
    V3[5]= denom*(-1j)*(-F1[2]*F2[4]-F1[5]*F2[3]+F1[3]*F2[5]+F1[4]*F2[2])
    return V3



def VVVV1P0_1(V2,V3,V4,COUP,M1,W1):
    V1 = wavefunctions.WaveFunction(size=6)
    V1[0] = +V2[0]+V3[0]+V4[0]
    V1[1] = +V2[1]+V3[1]+V4[1]
    P1 = [-complex(V1[0]).real, -complex(V1[1]).real, -complex(V1[1]).imag, -complex(V1[0]).imag]
    TMP1 = (V3[2]*V2[2]-V3[3]*V2[3]-V3[4]*V2[4]-V3[5]*V2[5])
    TMP2 = (V4[2]*V2[2]-V4[3]*V2[3]-V4[4]*V2[4]-V4[5]*V2[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1[2]= denom*(-1j*(V4[2]*TMP1)+1j*(V3[2]*TMP2))
    V1[3]= denom*(-1j*(V4[3]*TMP1)+1j*(V3[3]*TMP2))
    V1[4]= denom*(-1j*(V4[4]*TMP1)+1j*(V3[4]*TMP2))
    V1[5]= denom*(-1j*(V4[5]*TMP1)+1j*(V3[5]*TMP2))
    return V1



def VVVV3P0_1(V2,V3,V4,COUP,M1,W1):
    V1 = wavefunctions.WaveFunction(size=6)
    V1[0] = +V2[0]+V3[0]+V4[0]
    V1[1] = +V2[1]+V3[1]+V4[1]
    P1 = [-complex(V1[0]).real, -complex(V1[1]).real, -complex(V1[1]).imag, -complex(V1[0]).imag]
    TMP1 = (V3[2]*V2[2]-V3[3]*V2[3]-V3[4]*V2[4]-V3[5]*V2[5])
    TMP3 = (V3[2]*V4[2]-V3[3]*V4[3]-V3[4]*V4[4]-V3[5]*V4[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1[2]= denom*(-1j*(V4[2]*TMP1)+1j*(V2[2]*TMP3))
    V1[3]= denom*(-1j*(V4[3]*TMP1)+1j*(V2[3]*TMP3))
    V1[4]= denom*(-1j*(V4[4]*TMP1)+1j*(V2[4]*TMP3))
    V1[5]= denom*(-1j*(V4[5]*TMP1)+1j*(V2[5]*TMP3))
    return V1



def VVV1_0(V1,V2,V3,COUP):
    P1 = [complex(V1[0]).real, complex(V1[1]).real, complex(V1[1]).imag, complex(V1[0]).imag]
    P2 = [complex(V2[0]).real, complex(V2[1]).real, complex(V2[1]).imag, complex(V2[0]).imag]
    P3 = [complex(V3[0]).real, complex(V3[1]).real, complex(V3[1]).imag, complex(V3[0]).imag]
    TMP1 = (V3[2]*V2[2]-V3[3]*V2[3]-V3[4]*V2[4]-V3[5]*V2[5])
    TMP10 = (P2[0]*V1[2]-P2[1]*V1[3]-P2[2]*V1[4]-P2[3]*V1[5])
    TMP11 = (V1[2]*P3[0]-V1[3]*P3[1]-V1[4]*P3[2]-V1[5]*P3[3])
    TMP4 = (V2[2]*V1[2]-V2[3]*V1[3]-V2[4]*V1[4]-V2[5]*V1[5])
    TMP5 = (V3[2]*P1[0]-V3[3]*P1[1]-V3[4]*P1[2]-V3[5]*P1[3])
    TMP6 = (V3[2]*P2[0]-V3[3]*P2[1]-V3[4]*P2[2]-V3[5]*P2[3])
    TMP7 = (V3[2]*V1[2]-V3[3]*V1[3]-V3[4]*V1[4]-V3[5]*V1[5])
    TMP8 = (P1[0]*V2[2]-P1[1]*V2[3]-P1[2]*V2[4]-P1[3]*V2[5])
    TMP9 = (V2[2]*P3[0]-V2[3]*P3[1]-V2[4]*P3[2]-V2[5]*P3[3])
    vertex = COUP*(TMP1*(-1j*(TMP10)+1j*(TMP11))+(TMP4*(-1j*(TMP5)+1j*(TMP6))+TMP7*(+1j*(TMP8)-1j*(TMP9))))
    return vertex



def VVV1P0_1(V2,V3,COUP,M1,W1):
    P2 = [complex(V2[0]).real, complex(V2[1]).real, complex(V2[1]).imag, complex(V2[0]).imag]
    P3 = [complex(V3[0]).real, complex(V3[1]).real, complex(V3[1]).imag, complex(V3[0]).imag]
    V1 = wavefunctions.WaveFunction(size=6)
    V1[0] = +V2[0]+V3[0]
    V1[1] = +V2[1]+V3[1]
    P1 = [-complex(V1[0]).real, -complex(V1[1]).real, -complex(V1[1]).imag, -complex(V1[0]).imag]
    TMP1 = (V3[2]*V2[2]-V3[3]*V2[3]-V3[4]*V2[4]-V3[5]*V2[5])
    TMP5 = (V3[2]*P1[0]-V3[3]*P1[1]-V3[4]*P1[2]-V3[5]*P1[3])
    TMP6 = (V3[2]*P2[0]-V3[3]*P2[1]-V3[4]*P2[2]-V3[5]*P2[3])
    TMP8 = (P1[0]*V2[2]-P1[1]*V2[3]-P1[2]*V2[4]-P1[3]*V2[5])
    TMP9 = (V2[2]*P3[0]-V2[3]*P3[1]-V2[4]*P3[2]-V2[5]*P3[3])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1[2]= denom*(TMP1*(-1j*(P2[0])+1j*(P3[0]))+(V2[2]*(-1j*(TMP5)+1j*(TMP6))+V3[2]*(+1j*(TMP8)-1j*(TMP9))))
    V1[3]= denom*(TMP1*(-1j*(P2[1])+1j*(P3[1]))+(V2[3]*(-1j*(TMP5)+1j*(TMP6))+V3[3]*(+1j*(TMP8)-1j*(TMP9))))
    V1[4]= denom*(TMP1*(-1j*(P2[2])+1j*(P3[2]))+(V2[4]*(-1j*(TMP5)+1j*(TMP6))+V3[4]*(+1j*(TMP8)-1j*(TMP9))))
    V1[5]= denom*(TMP1*(-1j*(P2[3])+1j*(P3[3]))+(V2[5]*(-1j*(TMP5)+1j*(TMP6))+V3[5]*(+1j*(TMP8)-1j*(TMP9))))
    return V1



def VVVV4P0_1(V2,V3,V4,COUP,M1,W1):
    V1 = wavefunctions.WaveFunction(size=6)
    V1[0] = +V2[0]+V3[0]+V4[0]
    V1[1] = +V2[1]+V3[1]+V4[1]
    P1 = [-complex(V1[0]).real, -complex(V1[1]).real, -complex(V1[1]).imag, -complex(V1[0]).imag]
    TMP2 = (V4[2]*V2[2]-V4[3]*V2[3]-V4[4]*V2[4]-V4[5]*V2[5])
    TMP3 = (V3[2]*V4[2]-V3[3]*V4[3]-V3[4]*V4[4]-V3[5]*V4[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1[2]= denom*(-1j*(V3[2]*TMP2)+1j*(V2[2]*TMP3))
    V1[3]= denom*(-1j*(V3[3]*TMP2)+1j*(V2[3]*TMP3))
    V1[4]= denom*(-1j*(V3[4]*TMP2)+1j*(V2[4]*TMP3))
    V1[5]= denom*(-1j*(V3[5]*TMP2)+1j*(V2[5]*TMP3))
    return V1


