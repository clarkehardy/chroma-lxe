from chroma.geometry import Material, Solid, Surface
import numpy as np
#***************************************************************************
msuprasil = Material('msuprasil')
msuprasil.set('refractive_index', 1.57)
msuprasil.set('absorption_length', 1.152736998)
#msuprasil.set('absorption_length', .52736998)
msuprasil.set('scattering_length', 1e6)
msuprasil.density = 2.2
#***************************************************************************
teflon = Material('teflon')
teflon.set('refractive_index', 1.38)
teflon.set('absorption_length', 1)
teflon.set('scattering_length', 0)
teflon.density = 2.2
teflon.composition = {'F' : .9969, 'C' : .00063}
#***************************************************************************
steel= Material('steel')
steel.set('refractive_index', 1.07)
steel.set('absorption_length', 0)
steel.set('scattering_length', 0)
steel.density = 8.05
steel.composition = {'C' : .0008, 'Mg' : .02, 'P' : .0004, 'S' : .0003, 'Si' : .0075, 'Ni' : .08, 'Cr' : .18, 'Fe' : .711}
#***************************************************************************
copper= Material('copper')
copper.set('refractive_index', 1.3)
copper.set('absorption_length', 0)
copper.set('scattering_length',0)
copper.density = 8.96
copper.composition = {'Cu' : 1.00}
#***************************************************************************
ls = Material('ls')
ls.set('refractive_index', 1.5)
ls.set('absorption_length', 1e6)
ls.set('scattering_length', 1e6)
ls.density = 0.780
ls.composition = {'C' : .9, 'H' : .1}
#***************************************************************************
vacuum = Material('vac')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)
vacuum.density = 1
#***************************************************************************
lensmat = Material('lensmat')
lensmat.set('refractive_index', 2.0)
lensmat.set('absorption_length', 1e6)
lensmat.set('scattering_length', 1e6)
#***************************************************************************
lxe = Material('lxe')
lxe.set('refractive_index', 1.77) #according to https://www.sciencedirect.com/science/article/pii/S0168900203024331
lxe.set('absorption_length', 1E100) #according to https://arxiv.org/pdf/physics/0407033.pdf
lxe.set('scattering_length', 350.0) #according to figure 3.9 in https://pure.royalholloway.ac.uk/portal/files/29369028/2018GraceEPHD.pdf
lxe.density = 2.942  # according to https://userswww.pd.infn.it/~conti/images/LXe/density_vs_T.gif
#***************************************************************************
full_absorb = Material('full_absorb')
full_absorb.set('absorb', 1)
full_absorb.set('refractive_index', 1.5)
full_absorb.set('absorption_length', 1E100)
full_absorb.set('scattering_length', 1E100)
full_absorb.density = 1
#***************************************************************************
quartz = Material('quartz')
quartz.set('refractive_index', 1.6)
quartz.set('absorption_length', 9.49122)
quartz.set('scattering_length',1e6)
quartz.density = 2.65
#***************************************************************************
gold = Material('gold')
gold.set('refractive_index', 1.5215)    #according to https://refractiveindex.info/?shelf=main&book=Au&page=Werner
gold.set('absorption_length', 1e100)
gold.set('scattering_length',1e100)
gold.density = 19.32
#***************************************************************************
MgF2 = Material('MgF2')
MgF2.set('refractive_index', 1.44)    #according to http://www.esourceoptics.com/vuv_material_properties.html
MgF2.set('absorption_length',1e100)
MgF2.set('scattering_length',1e100)
MgF2.density = 3.15
#***************************************************************************
ceramic = Material('ceramic')
ceramic.set('refractive_index', 1.94)	#https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-o
ceramic.set('absorption_length',20)
ceramic.set('scattering_length',100)
ceramic.density = 3.15
#***************************************************************************
SiO2 = Material('SiO2')
SiO2.set('refractive_index', 1.97)
SiO2.set('absorption_length',100)
SiO2.set('scattering_length',100)
SiO2.density = 3.15
#***************************************************************************
