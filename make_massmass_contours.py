import sys,os
import numpy as np
import pickle
import json

from couplingscan.scan import *
from couplingscan.rescaler import *
from couplingscan.limitparsers import *
from basic_plotter import *

# Adjust to wherever you put the inputs
input_path = "../inputs/"

# Set up plot location for these ones
plot_path = "plots/validation"

# Couplings to test: you'll get a grid of all combinations
test_gq = [0.25,0.15,0.1,0.05]
test_gdm = [1.0,0.75,0.5,0.25]
test_gl = [0.1,0.05,0.01,0.0]

plotlims = (3500,1700)

# Begin main code
####################

# For cleaning out nan values
def clean_grid(xvals, yvals, zvals) :
  xclean = xvals[np.logical_not(np.logical_or(np.isnan(zvals),np.isinf(zvals)))]
  yclean = yvals[np.logical_not(np.logical_or(np.isnan(zvals),np.isinf(zvals)))]
  zclean = zvals[np.logical_not(np.logical_or(np.isnan(zvals),np.isinf(zvals)))]
  return xclean, yclean, zclean

# Now process everything.

# Grids to use for limits that start 1D.
# Big enough to stay valid for FCC
target_xvals = np.linspace(100,5000,50)
target_yvals = np.linspace(0,2000,101)
target_xgrid, target_ygrid = np.meshgrid(target_xvals,target_yvals)

# Get dijet data: we are beginning from gq limit for simplicity.
# Extract HEPData into useable format
with open("dijet_hepdata/hepdata_gqplot_cms36ifb.json", "r") as read_file:
  data = json.load(read_file)
invalues = data["values"]
# And convert to x-y numpy arrays
xlist_dijet = np.array([val["x"][0]["value"] for val in invalues]).astype(float)
ylist_dijet = np.array([val["y"][0]["value"] for val in invalues]).astype(float)

# Now create a 1d visible limit object with this, and extract our 2d limits from it.
# This is based on what the settings are for the world in which the gq plot is made.
gq_limit = CouplingLimit_Dijet(
    mmed=xlist_dijet,
    gq_limits=ylist_dijet,
    mdm=10000,
    gdm=0.0,
    gl=0.0,
    coupling='vector'
)
target_scan_A1 = DMAxialModelScan(mmed=target_xgrid.flatten(),mdm=target_ygrid.flatten(),
  gq=0.25, gdm=1.0, gl=0.0) 
dijet_exdepth_A1 = gq_limit.extract_exclusion_depths(target_scan_A1)

# Create A1 rescaler for dijets. 
rescaler_fromA1_dijet = Rescaler(target_scan_A1,dijet_exdepth_A1)

# Get monophoton data:
# Extract HEPData into useable format
with open("monophoton_hepdata/hepdata_AV_gq0p25_gchi1p0.json", "r") as read_file:
  data = json.load(read_file)
values = data["values"]

# Extract as numpy arrays
# Note: to run on monojet, swap the [1] and [0] in first two rows.
xlist_monophoton = np.array([val["x"][1]["value"] for val in values]).astype(float)
ylist_monophoton = np.array([val["x"][0]["value"] for val in values]).astype(float)
zlist_monophoton = np.array([val["y"][0]["value"] for val in values]).astype(float)

# Already a grid, so can go straight to equivalent scan with this one.
# It starts as A1, so we'll make our scan and rescaler from that.
monophoton_scan_A1 = DMAxialModelScan(mmed=xlist_monophoton,mdm=ylist_monophoton,
gq=0.25, gdm=1.0, gl=0.0)
rescaler_fromA1_monophotongrid = Rescaler(monophoton_scan_A1,zlist_monophoton)

# Collect a range of interesting axial model limits for both of these, including A2
dijet_exclusiondepths_axial = rescaler_fromA1_dijet.rescale_by_br_quarks(test_gq,test_gdm, test_gl,'axial')
monophoton_exclusiondepths_axial = rescaler_fromA1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'axial')

# Extract contours
monophoton_contours_axial = {}
dijet_contours_axial = {}
for coupling in monophoton_exclusiondepths_axial.keys() :
  monophoton_depth = monophoton_exclusiondepths_axial[coupling]
  xmono, ymono, zmono = clean_grid(xlist_monophoton, ylist_monophoton, monophoton_depth)
  monophoton_contours_axial[coupling] = get_contours(xmono, ymono, zmono)[0]
  dijet_depth = dijet_exclusiondepths_axial[coupling]
  xdij, ydij, zdij = clean_grid(target_scan_A1.mmed, target_scan_A1.mdm, dijet_depth)
  dijet_contours_axial[coupling] = get_contours(xdij, ydij, zdij)[0]

# Now let's do some vector scans.
# We can collect a range of interesting vector model scale factors, including V1 and V2.
dijet_exclusiondepths_vector = rescaler_fromA1_dijet.rescale_by_br_quarks(test_gq,test_gdm,test_gl,'vector')

# Recall we only want to convert mono-x limits between models once, since it's slow.
# So we'll go to V1 and then get other vector models from there.
# Get V1 limits: this is the slow bit
monophoton_limits_V1 = rescaler_fromA1_monophotongrid.rescale_by_hadronic_xsec_monox(0.25, 1.0, 0.0,'vector')[(0.25,1.0,0.0)]
# Scans and rescaler we'll use in monophoton
V1_scan_monophotongrid = DMVectorModelScan(mmed=xlist_monophoton, mdm=ylist_monophoton,gq=0.25, gdm=1.0, gl=0.0)
rescaler_fromV1_monophotongrid = Rescaler(V1_scan_monophotongrid,monophoton_limits_V1)
monophoton_exclusiondepths_vector = rescaler_fromV1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'vector')

# Extract contours
monophoton_contours_vector = {}
dijet_contours_vector = {}  
for coupling in monophoton_exclusiondepths_vector.keys() :
  monophoton_depth = monophoton_exclusiondepths_vector[coupling]
  print("Trying monophoton contours for",coupling,"vector")
  xmono, ymono, zmono = clean_grid(xlist_monophoton, ylist_monophoton, monophoton_depth)
  monophoton_contours_vector[coupling] = get_contours(xmono, ymono, zmono)[0]
  dijet_depth = dijet_exclusiondepths_vector[coupling]
  print("Trying dijet contours for",coupling,"vector")
  xdij, ydij, zdij = clean_grid(target_scan_A1.mmed, target_scan_A1.mdm, dijet_depth)
  dijet_contours_vector[coupling] = get_contours(xdij, ydij, zdij)[0]

# Save output in a clean way so that paper plot making script can be separate without re-running
with open("vector_exclusion_depths.pkl", "wb") as outfile_vec_depths :
  out_dict = {"dijet" : dijet_exclusiondepths_vector,
              "monophoton" : monophoton_exclusiondepths_vector}
  pickle.dump(out_dict, outfile_vec_depths)
with open("axial_exclusion_depths.pkl", "wb") as outfile_axial_depths :
  out_dict = {"dijet" : dijet_exclusiondepths_axial,
              "monophoton" : monophoton_exclusiondepths_axial}
  pickle.dump(out_dict, outfile_axial_depths)    
with open("vector_exclusion_contours.pkl", "wb") as poly_file:
  out_dict = {"dijet" : dijet_contours_vector,
              "monophoton" : monophoton_contours_vector}
  pickle.dump(out_dict, poly_file, pickle.HIGHEST_PROTOCOL)    
with open("axial_exclusion_contours.pkl", "wb") as poly_file:
  out_dict = {"dijet" : dijet_contours_axial,
              "monophoton" : monophoton_contours_axial}
  pickle.dump(out_dict, poly_file, pickle.HIGHEST_PROTOCOL)