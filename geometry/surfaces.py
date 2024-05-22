from chroma.geometry import Material, Solid, Surface
import numpy as np
#***************************************************************************
fulldetect = Surface('fulldetect')
fulldetect.set('detect', 1.0)
#***************************************************************************
SSuprasil = Surface('SSuprasil')
SSuprasil.set('detect', 0)
SSuprasil.set('absorb', 0)
SSuprasil.set('reflect_diffuse', 0)
SSuprasil.set('reflect_specular', 0)
SSuprasil.transmissive = 1.0
SSuprasil.thickness = 0.0
#***************************************************************************
teflon = Surface('teflon')
teflon.set('absorb', 0.1) #all values set to this before minute adjustmets
teflonReflArray = np.array([(175.0, 0.46), (260.0,  0.94),
              (270.0,  0.95), (280.0,  0.95), (290.0,  0.95), (300.0,  .95),
              (310.0, 0.96), (320.0, 0.98), (330.0, 0.98), (340.0, 0.98),
              (350.0, 0.98), (360.0, 0.99), (370.0, 0.99), (380.0, 0.99),
              (390.0, 0.99), (400.0, 0.99), (410.0, 0.99), (420.0, 0.99),
              (430.0, 0.99), (440.0, 0.99), (450.0, 0.99), (460.0, 0.99),
              (470.0, 0.99), (480.0, 0.99), (490.0, 0.99), (500.0, 0.99),
              (510.0, 0.99), (520.0, 0.99), (530.0, 0.99), (540.0, 0.99),
              (550.0, 0.99), (560.0,  0.99), (570.0,  0.99), (580.0,  0.99),
              (590.0,  0.99), (600.0,  0.99), (610.0,  0.99), (620.0,  0.99),
              (630.0, 0.99), (640.0,  0.99), (650.0,  0.99), (660.0,  0.99),
              (670.0,  0.99), (680.0,  0.99), (690.0,  0.99), (700.0,  0.99),
              (710.0,  0.99)])
teflonAbsorbArray = teflonReflArray
teflonAbsorbArray[:,1] = 1.00-teflonAbsorbArray[:,1]
# teflon.set('absorb', teflonAbsorbArray[:,1], teflonAbsorbArray[:,0])
teflon.set('absorb', 0.46)
teflon.set('reflect_diffuse', 0.97)              #according to https://arxiv.org/pdf/1612.07965.pdf
teflon.set('reflect_specular', 0.05)             #according to https://arxiv.org/pdf/0910.1056.pdf
#***************************************************************************
nothing = Surface('nothing')
nothing.set('detect', 0)
nothing.set('absorb', 0)
nothing.set('reflect_diffuse', 0)
nothing.set('reflect_specular', 0)
nothing.transmissive = 1
#***************************************************************************
quartz = Surface('quartz')
quartz.set('absorb', 0)
quartz.set('detect', 0)
#quartz.set('reflect_specular', 0)
#quartz.set('reflect_diffuse', 0)
#quartz.transmissive = 1
#***************************************************************************
gold = Surface('gold')
R = 0.29981
gold.set('absorb', 1-R)                          #unclear if this is necessary since the extinction coefficient k and the thickness are provided
gold.set('reflect_specular', R)                  #according to https://refractiveindex.info/?shelf=main&book=Au&page=Werner for an angle of 0 degrees
gold.set('eta', 0.0)
gold.set('k', 1.5186)                            #according to https://refractiveindex.info/?shelf=main&book=Au&page=Werner
gold.thickness = 0.001                           #need to verify this with Qidong
#***************************************************************************
MgF2 = Surface('MgF2')
# MgF2.set('absorb', 1.0)
MgF2.set('absorb', 0.0)
MgF2.set('reflect_diffuse', 0.0)
MgF2.set('reflect_specular', 1.0)
#***************************************************************************
steel = Surface('steel')                  #modified by Jacopo 07/31/2019
R = 0.12
steel.set('reflect_diffuse', R)		#https://www.klaran.com/images/kb/application-notes/Using-UV-Reflective-Materials-to-Maximize-Disinfection---Application-Note---AN011.pdf
steel.set('reflect_specular', R)
steel.set('absorb',1 - 2*R)
#***************************************************************************
ceramic = Surface('Ceramic')
ceramic_reflect = 0.35				#https://engineering.case.edu/centers/sdle/sites/engineering.case.edu.centers.sdle/files/optical_properties_of_aluminum_oxide_determined_f.pdf
ceramic.set('reflect_diffuse', ceramic_reflect)	#https://www.accuratus.com/accuflect/Accuflect_Reflectance.pdf
ceramic.set('absorb', 1 - ceramic_reflect)
#***************************************************************************
SiO2 = Surface('Si02')
SiO2_reflect = 0.3
QE = 0.21
SiO2.set('detect', QE*(1 - SiO2_reflect)/(1 + QE))
SiO2.set('absorb', (1 - SiO2_reflect)/(1 + QE))
SiO2.set('reflect_specular', SiO2_reflect)
#***************************************************************************
SiO2_off = Surface('Si02_off')
SiO2_off.set('detect', 0.0)
SiO2_off.set('absorb', 1 - SiO2_reflect)
SiO2_off.set('reflect_specular', SiO2_reflect)
#***************************************************************************
reflect0 = Surface('reflect0')
reflect0.set('absorb', 0.95)
reflect0.set('reflect_specular', 0.05)
#***************************************************************************
perfect_detector = Surface('perfect_detector')
perfect_detector.set('detect', 1.0)