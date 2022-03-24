import sys,os
import numpy as np
import pickle
import json

from couplingscan.scan import *
from couplingscan.rescaler import *
from couplingscan.limitparsers import *
from basic_plotter import *
from scipy import interpolate

# Adjust to wherever you put the inputs
input_path = "../inputs/"

# Set up plot location for these ones
plot_path = "plots/validation"

# Couplings to test: 
# turns out "go big or go home" does actually hit a limit here. So we're going to do a few versions of this.
# based on explicitly what we are trying to target
test_coupling_scenarios = {
  "gq_lim" : {
    "test_gq" : np.logspace(np.log10(0.01),0,101),
    "test_gdm" : [0.0, 1.0],
    "test_gl" : [0.0],
  },
  "gdm_lim" : {
    "test_gq" : [0.01, 0.1, 0.25],
    "test_gdm" : np.logspace(np.log10(0.01),0,101),
    "test_gl" : [0.0]
  },
  "gl_lim" : {
    "test_gq" : [0.01, 0.1, 0.25],
    "test_gdm" : [0.0, 1.0],
    "test_gl" : np.logspace(np.log10(0.01),0,101),
  }
}

# Should be able to use this for everything
mMed_test = np.linspace(0,5000,101)
test_mass_scenarios = {
  "dmDecoupled" : [10000 for i in mMed_test],
  "dmLight" : [1.0 for i in mMed_test],
  "DPLike" : [i/3.0 for i in mMed_test]
}

# plotlims = (3500,1700)

# Begin main code
####################

# For cleaning out nan values
def clean_grid(xvals, yvals, zvals) :
  xclean = xvals[np.isfinite(zvals)]
  yclean = yvals[np.isfinite(zvals)]
  zclean = zvals[np.isfinite(zvals)]

  return xclean, yclean, zclean

# Now process everything.
# We essentially want to 3d-ify our conversions and scans:
# going to get a hundred couplings and then slice through
# those in interpolated points along the DM, mDM values that we want.

# Grids to use for limits that start 1D.
# x value is mediator mass, y value is coupling.
# Want it evenly spaced in a log grid
target_mmed_vals = np.linspace(100,5000,50)
target_mdm_vals = np.linspace(0,2000,101)
target_mdm_vals = np.append(target_mdm_vals,10000)
target_gvals = np.logspace(np.log10(0.01),0,101)

target_xgrid, target_ygrid = np.meshgrid(target_mmed_vals,target_mdm_vals)

# Start by creating a mass-mass scan with baseline couplings and a rescaler. 
# We'll use this for dijets.
target_scan_A1 = DMAxialModelScan(mmed=target_xgrid.flatten(),mdm=target_ygrid.flatten(),
  gq=0.25, gdm=1.0, gl=0.0)  
rescaler_fromA1_dijetgrid = Rescaler(target_scan_A1)

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
dijet_exdepth_A1 = gq_limit.extract_exclusion_depths(target_scan_A1)

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
rescaler_fromA1_monophotongrid = Rescaler(monophoton_scan_A1)

# Storage
monophoton_contours_axial = {}
dijet_contours_axial = {}

# Now we're going to loop over our scenarios, going straight to contours and plots.
for test_scenario in test_coupling_scenarios.keys() :

  test_gq = test_coupling_scenarios[test_scenario]["test_gq"]
  test_gdm = test_coupling_scenarios[test_scenario]["test_gdm"]
  test_gl = test_coupling_scenarios[test_scenario]["test_gl"]

  # Collect a range of interesting axial model scale factors for both of these, including A2
  dijet_sfs_allaxial_fromA1 = rescaler_fromA1_dijetgrid.rescale_by_br_quarks(test_gq,test_gdm, test_gl,'axial')
  monophoton_sfs_allaxial_fromA1 = rescaler_fromA1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'axial')

  # Compute actual exclusion depths for axial
  monophoton_exclusiondepths_axial = {k : zlist_monophoton/v for k, v in monophoton_sfs_allaxial_fromA1.items()}
  dijet_exclusiondepths_axial = {k : dijet_exdepth_A1/v for k, v in dijet_sfs_allaxial_fromA1.items()}

  # For each test scenario, we actually have several sub-tests based on the grid of the non-scanned couplings.
  if "gq_lim" in test_scenario :
    test_couplings = test_gq
    coupling_index = 0
    other_first, other_second = np.meshgrid(test_gdm, test_gl)
    others_tag = "gdm{0}_gl{1}"
  elif "gdm_lim" in test_scenario :
    test_couplings = test_gdm
    coupling_index = 1
    other_first, other_second = np.meshgrid(test_gq, test_gl)
    others_tag = "gq{0}_gl{1}"
  else :
    test_couplings = test_gl
    coupling_index = 2
    other_first, other_second = np.meshgrid(test_gq, test_gdm)
    others_tag = "gq{0}_gdm{1}"
  for other_one, other_two in zip(other_first.flatten(), other_second.flatten()) :

    # Now need to extract contours. This will take interpolation: in addition to all the coupling scenarios
    # we have a few different slices we want to make through mMed, mDM space.
    for hypothesis in test_mass_scenarios.keys() :
      print("Beginning hypothesis",hypothesis)
      mDM_test = test_mass_scenarios[hypothesis]
      xvals = []
      yvals = []
      zvals_mono = []
      zvals_dijet = []
      # Monojet needs mirroring because it doesn't actually go to zero. Double everything.
      use_xvals_mono = np.concatenate((xlist_monophoton,xlist_monophoton))
      use_yvals_mono = np.concatenate((ylist_monophoton,-1.0*ylist_monophoton))
      for coupling in test_couplings :
        full_couplings = [other_one, other_two]
        full_couplings.insert(coupling_index, coupling)
        
        # Universal across analyses because we are doing the same interpolations
        xvals += list(mMed_test)
        yvals += [coupling for i in mMed_test]

        # Monojet extraction
        depths_mono = monophoton_exclusiondepths_axial[tuple(full_couplings)]
        zvals_mono_raw = interpolate.griddata((use_xvals_mono, use_yvals_mono), np.concatenate((depths_mono,depths_mono)),(mMed_test,mDM_test),method='linear')
        zvals_mono += list(zvals_mono_raw)

        # Dijet extraction
        depths_dijet = dijet_exclusiondepths_axial[tuple(full_couplings)]
        zvals_dijet_raw = interpolate.griddata((target_scan_A1.mmed, target_scan_A1.mdm), depths_dijet, (mMed_test, mDM_test),method='linear')
        zvals_dijet += list(zvals_dijet_raw)

      thiskey = "axial_{0}_{1}_".format(test_scenario,hypothesis)+others_tag.format(other_one, other_two)

      cleanx_mono, cleany_mono, cleanz_mono = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_mono))
      if cleanz_mono.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_mono, cleany_mono, cleanz_mono]], addPoints = False, this_tag = thiskey+"_monophoton",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        monophoton_contours_axial[thiskey] = get_contours(cleanx_mono, cleany_mono, cleanz_mono)[0]

      cleanx_dijet, cleany_dijet, cleanz_dijet = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_dijet))
      if cleanz_dijet.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_dijet, cleany_dijet, cleanz_dijet]], addPoints = False, this_tag = thiskey+"_dijet",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        monophoton_contours_axial[thiskey] = get_contours(cleanx_dijet, cleany_dijet, cleanz_dijet)[0]

print("Got here")
exit(1)


# Now let's do some vector scans.
# We can collect a range of interesting vector model scale factors, including V1 and V2.
dijet_sfs_allvector_fromA1 = rescaler_fromA1_dijetgrid.rescale_by_br_quarks(test_gq,test_gdm,test_gl,'vector')

# Recall we only want to convert mono-x limits between models once, since it's slow.
# So we'll go to V1 and then get other vector models from there.
# Scans and rescaler we'll use in monophoton
V1_scan_monophotongrid = DMVectorModelScan(mmed=xlist_monophoton, mdm=ylist_monophoton,gq=0.25, gdm=1.0, gl=0.0)
rescaler_fromV1_monophotongrid = Rescaler(V1_scan_monophotongrid)

# And the actual scale factors: this is the slow bit
monophoton_sfs_A1toV1 = rescaler_fromA1_monophotongrid.rescale_by_hadronic_xsec_monox(0.25, 1.0, 0.0,'vector')[(0.25,1.0,0.0)]

# Debug: this should match one of the plots that comes later.
debug_monophotonv1 = zlist_monophoton/monophoton_sfs_A1toV1
drawContourPlotRough([[xlist_monophoton, ylist_monophoton, debug_monophotonv1]], addPoints = False, this_tag = "monophoton_V1",plot_path = plot_path)

monophoton_sfs_allvector_fromV1 = rescaler_fromV1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'vector')

# Compute actual exclusion depths for both
dijet_exclusiondepths_vector = {k : dijet_exdepth_A1/v for k, v in dijet_sfs_allvector_fromA1.items()}
monophoton_exclusiondepths_vector = {k : zlist_monophoton/(monophoton_sfs_A1toV1*v) for k, v in monophoton_sfs_allvector_fromV1.items()}   

# Extract contours
monophoton_contours_vector = {}
dijet_contours_vector = {}  
for coupling in monophoton_exclusiondepths_vector.keys() :
  monophoton_depth = monophoton_exclusiondepths_vector[coupling]
  print("Trying monophoton contours for",coupling,"vector")
  monophoton_contours_vector[coupling] = get_contours(xlist_monophoton, ylist_monophoton, monophoton_depth)[0]
  dijet_depth = dijet_exclusiondepths_vector[coupling]
  print("Trying dijet contours for",coupling,"vector")
  dijet_contours_vector[coupling] = get_contours(target_scan_A1.mmed, target_scan_A1.mdm, dijet_depth)[0]

# Make some rough plots to validate everything 
for coupling in dijet_exclusiondepths_axial.keys() :
  grid_list_axial = [
    [xlist_monophoton, ylist_monophoton, monophoton_exclusiondepths_axial[coupling]],
    [target_scan_A1.mmed, target_scan_A1.mdm, dijet_exclusiondepths_axial[coupling]]
  ]
  grid_list_vector = [
    [xlist_monophoton, ylist_monophoton, monophoton_exclusiondepths_vector[coupling]],
    [target_scan_A1.mmed, target_scan_A1.mdm, dijet_exclusiondepths_vector[coupling]]
  ]

  drawContourPlotRough(grid_list_axial, addPoints = False, this_tag = "axial_gq{0}_gdm{1}_gl{2}".format(coupling[0],coupling[1],coupling[2]),plot_path = plot_path, xhigh=plotlims[0], yhigh=plotlims[1])
  drawContourPlotRough(grid_list_vector, addPoints = False, this_tag = "vector_gq{0}_gdm{1}_gl{2}".format(coupling[0],coupling[1],coupling[2]),plot_path = plot_path, xhigh=plotlims[0], yhigh=plotlims[1])

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