#  MadGraph5_aMC@NLO v. 3.5.7, 2024-11-29
#  By the MadGraph5_aMC@NLO Development Team
#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch

import cmath

class ParamCard(object):
    """ Accessor for a SLHA param card.dat."""

    def __init__(self, param_card_path=None):

        if param_card_path is not None:
            raise NotImplementedError(
                "The feature of loading independent parameter values "
                "from a param_card.dat file is not implemented yet.")

    def get_block_entry(self, block_name, entry_id, default_value):

        # In a future version we will retrieve this value from the param card.
        # For now simply always return the default value.
        return default_value

class ModelParameters(object):
    """ This class contains the list of parameters of a physics model sm and their definition."""

    def __init__(self, param_card=None):
        """ Instantiates using default value or the path of a SLHA param card."""
       
        # Param card accessor
        slha = ParamCard(param_card)
        
        self.ZERO = 0.

        # Computing independent parameters
        mdl_WH = slha.get_block_entry("decay", 25, 6.382339e-03);
        mdl_WT = slha.get_block_entry("decay", 6, 1.491500e+00);
        mdl_WW = slha.get_block_entry("decay", 24, 2.047600e+00);
        mdl_WZ = slha.get_block_entry("decay", 23, 2.441404e+00);
        mdl_MTA = slha.get_block_entry("mass", 15, 1.777000e+00);
        mdl_MH = slha.get_block_entry("mass", 25, 1.250000e+02);
        mdl_MB = slha.get_block_entry("mass", 5, 4.700000e+00);
        mdl_MT = slha.get_block_entry("mass", 6, 1.730000e+02);
        mdl_MZ = slha.get_block_entry("mass", 23, 9.118800e+01);
        mdl_ymtau = slha.get_block_entry("yukawa", 15, 1.777000e+00);
        mdl_ymt = slha.get_block_entry("yukawa", 6, 1.730000e+02);
        mdl_ymb = slha.get_block_entry("yukawa", 5, 4.700000e+00);
        aS = slha.get_block_entry("sminputs", 3, 1.180000e-01);
        mdl_Gf = slha.get_block_entry("sminputs", 2, 1.166390e-05);
        aEWM1 = slha.get_block_entry("sminputs", 1, 1.325070e+02);
        mdl_conjg__CKM1x1 = 1.0
        mdl_conjg__CKM3x3 = 1.0
        mdl_CKM3x3 = 1.0
        mdl_complexi = complex(0,1)
        mdl_MZ__exp__2 = mdl_MZ**2
        mdl_MZ__exp__4 = mdl_MZ**4
        mdl_sqrt__2 =  cmath.sqrt(2) 
        mdl_MH__exp__2 = mdl_MH**2
        mdl_aEW = 1/aEWM1
        mdl_MW = cmath.sqrt(mdl_MZ__exp__2/2. + cmath.sqrt(mdl_MZ__exp__4/4. - (mdl_aEW*cmath.pi*mdl_MZ__exp__2)/(mdl_Gf*mdl_sqrt__2)))
        mdl_sqrt__aEW =  cmath.sqrt(mdl_aEW) 
        mdl_ee = 2*mdl_sqrt__aEW*cmath.sqrt(cmath.pi)
        mdl_MW__exp__2 = mdl_MW**2
        mdl_sw2 = 1 - mdl_MW__exp__2/mdl_MZ__exp__2
        mdl_cw = cmath.sqrt(1 - mdl_sw2)
        mdl_sqrt__sw2 =  cmath.sqrt(mdl_sw2) 
        mdl_sw = mdl_sqrt__sw2
        mdl_g1 = mdl_ee/mdl_cw
        mdl_gw = mdl_ee/mdl_sw
        mdl_vev = (2*mdl_MW*mdl_sw)/mdl_ee
        mdl_vev__exp__2 = mdl_vev**2
        mdl_lam = mdl_MH__exp__2/(2.*mdl_vev__exp__2)
        mdl_yb = (mdl_ymb*mdl_sqrt__2)/mdl_vev
        mdl_yt = (mdl_ymt*mdl_sqrt__2)/mdl_vev
        mdl_ytau = (mdl_ymtau*mdl_sqrt__2)/mdl_vev
        mdl_muH = cmath.sqrt(mdl_lam*mdl_vev__exp__2)
        mdl_I1x33 = mdl_yb*mdl_conjg__CKM3x3
        mdl_I2x33 = mdl_yt*mdl_conjg__CKM3x3
        mdl_I3x33 = mdl_CKM3x3*mdl_yt
        mdl_I4x33 = mdl_CKM3x3*mdl_yb
        mdl_ee__exp__2 = mdl_ee**2
        mdl_sw__exp__2 = mdl_sw**2
        mdl_cw__exp__2 = mdl_cw**2

        # Computing independent couplings


        # Computing dependent parameters
        mdl_sqrt__aS =  cmath.sqrt(aS) 
        G = 2*mdl_sqrt__aS*cmath.sqrt(cmath.pi)
        mdl_G__exp__2 = G**2

        # Computing independent parameters
        GC_10 = -G
        GC_11 = mdl_complexi*G
        GC_12 = mdl_complexi*mdl_G__exp__2

        # Setting independent parameters
        # Model parameters independent of aS
        self.mdl_WH = float(mdl_WH.real)
        self.mdl_WT = float(mdl_WT.real)
        self.mdl_WW = float(mdl_WW.real)
        self.mdl_WZ = float(mdl_WZ.real)
        self.mdl_MTA = float(mdl_MTA.real)
        self.mdl_MH = float(mdl_MH.real)
        self.mdl_MB = float(mdl_MB.real)
        self.mdl_MT = float(mdl_MT.real)
        self.mdl_MZ = float(mdl_MZ.real)
        self.mdl_ymtau = float(mdl_ymtau.real)
        self.mdl_ymt = float(mdl_ymt.real)
        self.mdl_ymb = float(mdl_ymb.real)
        self.aS = float(aS.real)
        self.mdl_Gf = float(mdl_Gf.real)
        self.aEWM1 = float(aEWM1.real)
        self.mdl_conjg__CKM1x1 = float(mdl_conjg__CKM1x1.real)
        self.mdl_conjg__CKM3x3 = float(mdl_conjg__CKM3x3.real)
        self.mdl_CKM3x3 = float(mdl_CKM3x3.real)
        self.mdl_complexi = complex(mdl_complexi)
        self.mdl_MZ__exp__2 = float(mdl_MZ__exp__2.real)
        self.mdl_MZ__exp__4 = float(mdl_MZ__exp__4.real)
        self.mdl_sqrt__2 = float(mdl_sqrt__2.real)
        self.mdl_MH__exp__2 = float(mdl_MH__exp__2.real)
        self.mdl_aEW = float(mdl_aEW.real)
        self.mdl_MW = float(mdl_MW.real)
        self.mdl_sqrt__aEW = float(mdl_sqrt__aEW.real)
        self.mdl_ee = float(mdl_ee.real)
        self.mdl_MW__exp__2 = float(mdl_MW__exp__2.real)
        self.mdl_sw2 = float(mdl_sw2.real)
        self.mdl_cw = float(mdl_cw.real)
        self.mdl_sqrt__sw2 = float(mdl_sqrt__sw2.real)
        self.mdl_sw = float(mdl_sw.real)
        self.mdl_g1 = float(mdl_g1.real)
        self.mdl_gw = float(mdl_gw.real)
        self.mdl_vev = float(mdl_vev.real)
        self.mdl_vev__exp__2 = float(mdl_vev__exp__2.real)
        self.mdl_lam = float(mdl_lam.real)
        self.mdl_yb = float(mdl_yb.real)
        self.mdl_yt = float(mdl_yt.real)
        self.mdl_ytau = float(mdl_ytau.real)
        self.mdl_muH = float(mdl_muH.real)
        self.mdl_I1x33 = complex(mdl_I1x33)
        self.mdl_I2x33 = complex(mdl_I2x33)
        self.mdl_I3x33 = complex(mdl_I3x33)
        self.mdl_I4x33 = complex(mdl_I4x33)
        self.mdl_ee__exp__2 = float(mdl_ee__exp__2.real)
        self.mdl_sw__exp__2 = float(mdl_sw__exp__2.real)
        self.mdl_cw__exp__2 = float(mdl_cw__exp__2.real)

        # Setting independent couplings
        # Model parameters dependent on aS
        self.mdl_sqrt__aS = float(mdl_sqrt__aS.real)
        self.G = float(G.real)
        self.mdl_G__exp__2 = float(mdl_G__exp__2.real)

        # Setting dependent parameters
        # Model couplings independent of aS


        # Setting independent parameters
        # Model couplings dependent on aS
        self.GC_10 = complex(GC_10)
        self.GC_11 = complex(GC_11)
        self.GC_12 = complex(GC_12)

    def __str__(self):
        """ Print all parameters contained in this model."""
    
        res = ['>>> Model sm <<<']
        res.append('')
        res.append('Independent parameters:')
        res.append('-----------------------')
        res.append('')
        res.append('{:<20s} = {:<20.16e}'.format('mdl_WH',self.mdl_WH))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_WT',self.mdl_WT))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_WW',self.mdl_WW))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_WZ',self.mdl_WZ))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MTA',self.mdl_MTA))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MH',self.mdl_MH))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MB',self.mdl_MB))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MT',self.mdl_MT))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MZ',self.mdl_MZ))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ymtau',self.mdl_ymtau))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ymt',self.mdl_ymt))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ymb',self.mdl_ymb))
        res.append('{:<20s} = {:<20.16e}'.format('aS',self.aS))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_Gf',self.mdl_Gf))
        res.append('{:<20s} = {:<20.16e}'.format('aEWM1',self.aEWM1))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_conjg__CKM1x1',self.mdl_conjg__CKM1x1))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_conjg__CKM3x3',self.mdl_conjg__CKM3x3))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_CKM3x3',self.mdl_CKM3x3))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_complexi',self.mdl_complexi))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MZ__exp__2',self.mdl_MZ__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MZ__exp__4',self.mdl_MZ__exp__4))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sqrt__2',self.mdl_sqrt__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MH__exp__2',self.mdl_MH__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_aEW',self.mdl_aEW))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MW',self.mdl_MW))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sqrt__aEW',self.mdl_sqrt__aEW))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ee',self.mdl_ee))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_MW__exp__2',self.mdl_MW__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sw2',self.mdl_sw2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_cw',self.mdl_cw))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sqrt__sw2',self.mdl_sqrt__sw2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sw',self.mdl_sw))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_g1',self.mdl_g1))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_gw',self.mdl_gw))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_vev',self.mdl_vev))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_vev__exp__2',self.mdl_vev__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_lam',self.mdl_lam))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_yb',self.mdl_yb))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_yt',self.mdl_yt))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ytau',self.mdl_ytau))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_muH',self.mdl_muH))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_I1x33',self.mdl_I1x33))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_I2x33',self.mdl_I2x33))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_I3x33',self.mdl_I3x33))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_I4x33',self.mdl_I4x33))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_ee__exp__2',self.mdl_ee__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sw__exp__2',self.mdl_sw__exp__2))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_cw__exp__2',self.mdl_cw__exp__2))

        res.append('')
        res.append('Independent couplings:')
        res.append('----------------------')
        res.append('')


        res.append('')
        res.append('Dependent parameters:')
        res.append('---------------------')
        res.append('')
        res.append('{:<20s} = {:<20.16e}'.format('mdl_sqrt__aS',self.mdl_sqrt__aS))
        res.append('{:<20s} = {:<20.16e}'.format('G',self.G))
        res.append('{:<20s} = {:<20.16e}'.format('mdl_G__exp__2',self.mdl_G__exp__2))

        res.append('')
        res.append('Dependent couplings:')
        res.append('--------------------')
        res.append('')
        res.append('{:<20s} = {:<20.16e}'.format('GC_10',self.GC_10))
        res.append('{:<20s} = {:<20.16e}'.format('GC_11',self.GC_11))
        res.append('{:<20s} = {:<20.16e}'.format('GC_12',self.GC_12))

        res.append('')


        return '\n'.join(res)


