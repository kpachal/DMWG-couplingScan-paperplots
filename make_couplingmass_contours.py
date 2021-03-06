import sys,os
import numpy as np
import pickle
import json

from couplingscan.scan import *
from couplingscan.rescaler import *
from couplingscan.limitparsers import *
from basic_plotter import *
from scipy import interpolate
import ROOT

# Adjust to wherever you put the inputs
input_path = "../inputs/"

# Set up plot location for these ones
plot_path = "plots/validation"

# Couplings to test: 
# turns out "go big or go home" does actually hit a limit here. So we're going to do a few versions of this.
# based on explicitly what we are trying to target
test_coupling_scenarios = {
  "gq_lim" : {
    "test_gq" : np.logspace(np.log10(0.001),0,101),
    "test_gdm" : [0.0, 1.0],
    "test_gl" : [0.0],
  },
  "gdm_lim" : {
    "test_gq" : [0.01, 0.1, 0.25],
    "test_gdm" : np.logspace(np.log10(0.001),0,101),
    "test_gl" : [0.0]
  },
  "gl_lim" : {
    "test_gq" : [0.01, 0.1, 0.25],
    "test_gdm" : [0.0, 1.0],
    "test_gl" : np.logspace(np.log10(0.001),0,101),
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

# Start by creating a mass-mass scan with baseline couplings. 
# We'll use this for dijets.
target_scan_A2 = DMAxialModelScan(mmed=target_xgrid.flatten(),mdm=target_ygrid.flatten(),
  gq=0.25, gdm=1.0, gl=0.1)  

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
dijet_exdepth_A2 = gq_limit.extract_exclusion_depths(target_scan_A2)

rescaler_fromA2_dijetgrid = Rescaler(target_scan_A2, dijet_exdepth_A2)

# Get dilepton data:
# Extract HEPData into useable format
with open("dilepton_data/hepdata_observed_xseclimits_atlas139ifb.json", "r") as read_file:
  data = json.load(read_file)
values = data["values"]
widths = data["qualifiers"]["RELATIVE WIDTH"]
# And convert to x-y numpy arrays
xlist_dilepton = np.array([val["x"][0]["value"] for val in values]).astype(float)
ylists = {}
for group in range(6) :
  ylist = np.array([val["y"][group]["value"] for val in values]).astype(float)
  width = widths[group]["value"]
  float_width = float(width.replace(" %",""))
  frac_width = float_width/100
  ylists[frac_width] = ylist

# Read in our theory curve - very approximate but will work for this test
xvals_th_dil = []
yvals_th_dil = []
with open("dilepton_data/approximate_theorycurve.txt", "r") as read_file:
  lines = read_file.readlines()
  for line in lines :
    tokens = line.split(", ")
    xvals_th_dil.append(1000*float(tokens[0])) # this was in TeV
    yvals_th_dil.append(float(tokens[1]))
x_theory_dilepton = np.array(xvals_th_dil)
y_theory_dilepton = np.array(yvals_th_dil)

# Now create a visible limit object with this, and extract our 2d limits from it.
dilepton_limit = CrossSectionLimit_Dilepton(
    mmed_limit=xlist_dilepton,
    xsec_limit=ylists,
    mmed_theory=x_theory_dilepton,
    xsec_theory=y_theory_dilepton,
    mdm=2.5,
    gq=0.1,
    gdm=1.0,
    gl=0.01,
    coupling='vector'
)
dilepton_exdepths_A2 = dilepton_limit.extract_exclusion_depths(target_scan_A2)
dilepton_exdepth_A2 = dilepton_limit.select_depths(target_scan_A2,dilepton_exdepths_A2)

# Dilepton rescaler
rescaler_fromA2_dilepton = Rescaler(target_scan_A2,dilepton_exdepths_A2)

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
rescaler_fromA1_monophotongrid = Rescaler(monophoton_scan_A1, zlist_monophoton, 10.0)

# Recall we only want to convert mono-x limits between models once, since it's slow.
# So we'll go from A1 to V1 and then get other vector models from there.
# This is the slow bit
monophoton_limits_V1 = rescaler_fromA1_monophotongrid.rescale_by_hadronic_xsec_monox(0.25, 1.0, 0.0,'vector')[(0.25,1.0,0.0)]
# TESTS ONLY
#monophoton_limits_V1 = rescaler_fromA1_monophotongrid.rescale_by_parton_level_xsec_monox(0.25, 1.0, 0.0,'vector')[(0.25,1.0,0.0)]

# Scans and rescaler from here
V1_scan_monophotongrid = DMVectorModelScan(mmed=xlist_monophoton, mdm=ylist_monophoton,gq=0.25, gdm=1.0, gl=0.0)
rescaler_fromV1_monophotongrid = Rescaler(V1_scan_monophotongrid, monophoton_limits_V1, 10.0)

# Storage
monophoton_contours_axial = {}
dijet_contours_axial = {}
dilepton_contours_axial = {}
monophoton_contours_vector = {}
dijet_contours_vector = {}
dilepton_contours_vector = {}

# Now we're going to loop over our scenarios, going straight to contours and plots.
for test_scenario in test_coupling_scenarios.keys() :

  test_gq = test_coupling_scenarios[test_scenario]["test_gq"]
  test_gdm = test_coupling_scenarios[test_scenario]["test_gdm"]
  test_gl = test_coupling_scenarios[test_scenario]["test_gl"]

  # Collect a range of interesting axial model scale factors for both of these, including A2
  dijet_exclusiondepths_axial = rescaler_fromA2_dijetgrid.rescale_by_br_quarks(test_gq,test_gdm, test_gl,'axial')
  monophoton_exclusiondepths_axial = rescaler_fromA1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'axial')
  #dilepton_exclusiondepths_axial = rescaler_fromA2_dijetgrid.rescale_by_br_leptons(test_gq,test_gdm, test_gl, 'axial')

  # And some vector scans.
  dijet_exclusiondepths_vector = rescaler_fromA2_dijetgrid.rescale_by_br_quarks(test_gq,test_gdm,test_gl,'vector')
  monophoton_exclusiondepths_vector = rescaler_fromV1_monophotongrid.rescale_by_propagator(test_gq,test_gdm,test_gl,'vector')
  #dilepton_exclusiondepths_vector = rescaler_fromA2_dijetgrid.rescale_by_br_leptons(test_gq,test_gdm,test_gl,'vector')

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
      zvals_mono_axial = []
      zvals_dijet_axial = []
      zvals_dilep_axial = []
      zvals_mono_vector = []
      zvals_dijet_vector = []      
      zvals_dilep_vector = []
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
        depths_mono_axial = monophoton_exclusiondepths_axial[tuple(full_couplings)]
        zvals_mono_axial_raw = interpolate.griddata((use_xvals_mono, use_yvals_mono), np.concatenate((depths_mono_axial,depths_mono_axial)),(mMed_test,mDM_test),method='linear')
        zvals_mono_axial += list(zvals_mono_axial_raw)
        depths_mono_vector = monophoton_exclusiondepths_vector[tuple(full_couplings)]
        zvals_mono_vector_raw = interpolate.griddata((use_xvals_mono, use_yvals_mono), np.concatenate((depths_mono_vector,depths_mono_vector)),(mMed_test,mDM_test),method='linear')
        zvals_mono_vector += list(zvals_mono_vector_raw)

        # Dijet extraction
        depths_dijet_axial = dijet_exclusiondepths_axial[tuple(full_couplings)]
        zvals_dijet_axial_raw = interpolate.griddata((target_scan_A2.mmed, target_scan_A2.mdm), depths_dijet_axial, (mMed_test, mDM_test),method='linear')
        zvals_dijet_axial += list(zvals_dijet_axial_raw)
        depths_dijet_vector = dijet_exclusiondepths_vector[tuple(full_couplings)]
        zvals_dijet_vector_raw = interpolate.griddata((target_scan_A2.mmed, target_scan_A2.mdm), depths_dijet_vector, (mMed_test, mDM_test),method='linear')
        zvals_dijet_vector += list(zvals_dijet_vector_raw)

        # Dilepton extraction
        depths_dilepton_axial = dilepton_exclusiondepths_axial[tuple(full_couplings)]
        zvals_dilep_axial_raw = interpolate.griddata((target_scan_A2.mmed, target_scan_A2.mdm), depths_dilepton_axial, (mMed_test, mDM_test),method='linear')
        zvals_dilep_axial += list(zvals_dilep_axial_raw)
        depths_dilepton_vector = dilepton_exclusiondepths_vector[tuple(full_couplings)]
        zvals_dilep_vector_raw = interpolate.griddata((target_scan_A2.mmed, target_scan_A2.mdm), depths_dilepton_vector, (mMed_test, mDM_test),method='linear')
        zvals_dilep_vector += list(zvals_dilep_axial_raw)

      thiskey = "axial_{0}_{1}_".format(test_scenario,hypothesis)+others_tag.format(other_one, other_two)

      cleanx_mono, cleany_mono, cleanz_mono = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_mono_axial))
      if cleanz_mono.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_mono, cleany_mono, cleanz_mono]], addPoints = False, this_tag = thiskey+"_monophoton",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        monophoton_contours_axial[thiskey] = get_contours(cleanx_mono, cleany_mono, cleanz_mono)[0]

      cleanx_dijet, cleany_dijet, cleanz_dijet = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_dijet_axial))
      if cleanz_dijet.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_dijet, cleany_dijet, cleanz_dijet]], addPoints = False, this_tag = thiskey+"_dijet",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        dijet_contours_axial[thiskey] = get_contours(cleanx_dijet, cleany_dijet, cleanz_dijet)[0]

      cleanx_dilep, cleany_dilep, cleanz_dilep = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_dilep_axial))
      if cleanz_dilep.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_dilep, cleany_dilep, cleanz_dilep]], addPoints = False, this_tag = thiskey+"_dilepton",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        dilepton_contours_axial[thiskey] = get_contours(cleanx_dilep, cleany_dilep, cleanz_dilep)[0]        

      thiskey = "vector_{0}_{1}_".format(test_scenario,hypothesis)+others_tag.format(other_one, other_two)

      cleanx_mono, cleany_mono, cleanz_mono = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_mono_vector))
      if cleanz_mono.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_mono, cleany_mono, cleanz_mono]], addPoints = False, this_tag = thiskey+"_monophoton",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        monophoton_contours_vector[thiskey] = get_contours(cleanx_mono, cleany_mono, cleanz_mono)[0]

      cleanx_dijet, cleany_dijet, cleanz_dijet = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_dijet_vector))
      if cleanz_dijet.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_dijet, cleany_dijet, cleanz_dijet]], addPoints = False, this_tag = thiskey+"_dijet",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        dijet_contours_vector[thiskey] = get_contours(cleanx_dijet, cleany_dijet, cleanz_dijet)[0]

      cleanx_dilep, cleany_dilep, cleanz_dilep = clean_grid(np.array(xvals), np.array(yvals), np.array(zvals_dilep_vector))
      if cleanz_dilep.size > 0 :
        # Make a quick plot to check this looks sane
        drawContourPlotRough([[cleanx_dilep, cleany_dilep, cleanz_dilep]], addPoints = False, this_tag = thiskey+"_dilepton",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)
        dilepton_contours_vector[thiskey] = get_contours(cleanx_dilep, cleany_dilep, cleanz_dilep)[0]        

# Save outputs in a clean way so that plot making script can be separate without re-running.
# Also save some TGraphs for easy cross checks.
with open("vector_exclusion_depths_couplingmass.pkl", "wb") as outfile_vec_depths :
  out_dict = {"dijet" : dijet_exclusiondepths_vector,
              "monophoton" : monophoton_exclusiondepths_vector,
              "dilepton" : dilepton_exclusiondepths_vector}
  pickle.dump(out_dict, outfile_vec_depths)
with open("axial_exclusion_depths_couplingmass.pkl", "wb") as outfile_axial_depths :
  out_dict = {"dijet" : dijet_exclusiondepths_axial,
              "monophoton" : monophoton_exclusiondepths_axial,
              "dilepton" : dilepton_exclusiondepths_axial}
  pickle.dump(out_dict, outfile_axial_depths)    
with open("vector_exclusion_contours_couplingmass.pkl", "wb") as poly_file:
  out_dict = {"dijet" : dijet_contours_vector,
              "monophoton" : monophoton_contours_vector,
              "dilepton" : dilepton_contours_vector}
  pickle.dump(out_dict, poly_file, pickle.HIGHEST_PROTOCOL)    
with open("axial_exclusion_contours_couplingmass.pkl", "wb") as poly_file:
  out_dict = {"dijet" : dijet_contours_axial,
              "monophoton" : monophoton_contours_axial,
              "dilepton" : dilepton_contours_axial}
  pickle.dump(out_dict, poly_file, pickle.HIGHEST_PROTOCOL)

# And some TGraphs for ease
big_ol_dict = {
  "vector" : {"dijet" : dijet_contours_vector,
              "monophoton" : monophoton_contours_vector,
              "dilepton" : dilepton_contours_vector},
  "axial" : {"dijet" : dijet_contours_axial,
              "monophoton" : monophoton_contours_axial,
              "dilepton" : dilepton_contours_axial}
}
for model, middict in big_ol_dict.items() :
  outfile = ROOT.TFile.Open("{0}_exclusion_contours_couplingmass.root".format(model), "RECREATE")
  outfile.cd()    
  for analysis, smalldict in middict.items() :
    for key, contour in smalldict.items() :
      igraph = ROOT.TGraph()
      for icontour in contour :
        for x, y in list(icontour.exterior.coords) :
          igraph.AddPoint(x,y)
      outname = "{0}_{1}".format(key,analysis)
      igraph.Write(outname)
outfile.Close()

# And finally save original one for comparison
outfile_orig = ROOT.TFile.Open("original_dijet_gqlim.root","RECREATE")
outfile_orig.cd()
igraph = ROOT.TGraph()
for x, y in zip(xlist_dijet,ylist_dijet) :
  igraph.AddPoint(x,y)
igraph.Write("gq_original")
outfile_orig.Close()
