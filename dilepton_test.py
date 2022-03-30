import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ROOT

from basic_plotter import *
from couplingscan.scan import *
from couplingscan.rescaler import *
from couplingscan.limitparsers import *

# Analysing results from http://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-16-056/

plot_tag = ""

analysis_tag = "ATLAS-dilepton-internal"
plot_path = "plots/validation"

def get_aspect_ratio(ax) :
  ratio = 1.0
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  return abs((xright-xleft)/(ybottom-ytop))*ratio

def make_xsec_plot(xvals_obs, yvals_obs, xvals_th, yvals_th, this_tag) :
  plt.clf()
  plt.plot(xvals_obs,yvals_obs,label="observed")
  plt.plot(xvals_th,yvals_th,label="theory")
  plt.legend()
  plt.savefig('plots/{0}_{1}.pdf'.format(analysis_tag,this_tag),bbox_inches='tight')

def clean_grid(xvals, yvals, zvals) :
  if zvals.size == 0 :
    return np.array([]), np.array([]), np.array([])
  xclean = xvals[np.logical_and(np.logical_and(np.isfinite(xvals),np.isfinite(yvals)),np.isfinite(zvals))]
  yclean = yvals[np.logical_and(np.logical_and(np.isfinite(xvals),np.isfinite(yvals)),np.isfinite(zvals))]
  zclean = zvals[np.logical_and(np.logical_and(np.isfinite(xvals),np.isfinite(yvals)),np.isfinite(zvals))]
  return xclean, yclean, zclean  

def make_plot(xvals, yvals, zvals, this_tag, addText=None, addCurves=None, addPoints=False) :

  levels = range(26)  # Levels must be increasing.
  fig,ax=plt.subplots(1,1)
  plt.xlim(0, 3500)
  plt.ylim(0, 1700)
  plt.rc('font',size=17)
  ratio = get_aspect_ratio(ax)
  ax.set_aspect(ratio)
  cp = ax.tricontourf(xvals, yvals, zvals, levels=levels, cmap='Blues_r')
  fig.colorbar(cp)

  # Want points under contour, if adding them.
  if addPoints :
    # Separate into two populations: excluded and non excluded.
    xexcl,yexcl = [],[]
    xnon,ynon = [],[]
    for x,y,z in zip(xvals,yvals,zvals) :
      if z < 1. : 
        xexcl.append(x)
        yexcl.append(y)
      else :
        xnon.append(x)
        ynon.append(y)
    #for i, j, k in zip(xvals,yvals,zvals) :
    ax.scatter(xnon,ynon,color='red', marker='o',facecolors='none',linewidths=1,s=1)
    ax.scatter(xexcl,yexcl,color='white', marker='o',facecolors='none',linewidths=1,s=1)

  ax.set_xlabel("m$_{ZA}$ [GeV]")
  ax.set_ylabel("m$_{\chi}$ [GeV]")   

  # Now add exclusion contour (if not doing official - harder to see with both)
  if not addCurves :
    ax.tricontour(xvals, yvals, zvals,levels=[1],colors=['w'],linewidths=[2])

  # Now add another for comparison if desired.
  if addCurves :
    for curve in addCurves :
      ax.add_patch(curve)

  # Add text
  if addText :
    plt.figtext(0.2,0.75,addText,size=14)
    #plt.figtext(0.2,0.75,addText,backgroundcolor="white",size=14)

  plt.savefig('plots/{0}_{1}.eps'.format(analysis_tag,this_tag),bbox_inches='tight')
  plt.savefig('plots/{0}_{1}.pdf'.format(analysis_tag,this_tag),bbox_inches='tight')

# Create scan beginning from multiple observed limits of different widths
# and a cross section limit (approximate here).
# Extract HEPData into useable format
with open("dilepton_data/hepdata_observed_xseclimits_atlas139ifb.json", "r") as read_file:
  data = json.load(read_file)
values = data["values"]
widths = data["qualifiers"]["RELATIVE WIDTH"]
# And convert to x-y numpy arrays
xlist = np.array([val["x"][0]["value"] for val in values]).astype(float)
ylists = {}
for group in range(6) :
  ylist = np.array([val["y"][group]["value"] for val in values]).astype(float)
  width = widths[group]["value"]
  float_width = float(width.replace(" %",""))
  frac_width = float_width/100
  ylists[frac_width] = ylist

# Read in our theory curve - very approximate but will work for this test
xvals = []
yvals = []
with open("dilepton_data/approximate_theorycurve.txt", "r") as read_file:
  lines = read_file.readlines()
  for line in lines :
    tokens = line.split(", ")
    xvals.append(1000*float(tokens[0])) # this was in TeV
    yvals.append(float(tokens[1]))
x_theory = np.array(xvals)
y_theory = np.array(yvals)

# Now create a visible limit object with this, and extract our 2d limits from it.
# We will give it a full set of observed limits.
# When we do this, we imply that larger intrinsic widths than those passed to it are
# not valid to exclude with this analysis.
# If only one observed limit is given, we treat it as valid everywhere and leave it
# to the user to decide when to cut it off.
dilepton_limit = CrossSectionLimit_Dilepton(
    mmed_limit=xlist,
    xsec_limit=ylists,
    mmed_theory=x_theory,
    xsec_theory=y_theory,
    mdm=2.5,
    gq=0.1,
    gdm=1.0,
    gl=0.01,
    coupling='vector'
)

# Plot to validate.
make_xsec_plot(xlist,ylists[0.03],x_theory,y_theory,"input")

# A1 and V1 have no lepton coupling so we don't need to worry about them. 
# We'll extract directly to A2 and V2 from our scan.
target_xvals = np.linspace(300,2000,171)
target_yvals = np.linspace(0,1700,35)
target_xgrid, target_ygrid = np.meshgrid(target_xvals,target_yvals)

# Let's start with going straight to V2 since it's literally the same thing.
# The input should be one line through the V2 curve.
scan_V2 = DMVectorModelScan(
mmed=target_xgrid.flatten(),
mdm=target_ygrid.flatten(),
gq=0.1,
gdm=1.0,
gl=0.01,
)
all_depths_V2 = dilepton_limit.extract_exclusion_depths(scan_V2)
values_V2 = dilepton_limit.select_depths(scan_V2,all_depths_V2)
x, y, z = clean_grid(scan_V2.mmed, scan_V2.mdm, values_V2)
make_plot(x, y, z, "V2_direct", addText=None, addCurves=None, addPoints=True)

# Make an A2 scan
scan_A2 = DMAxialModelScan(
mmed=target_xgrid.flatten(),
mdm=target_ygrid.flatten(),
gq=0.1,
gdm=1.0,
gl=0.1,
)

all_depths_A2 = dilepton_limit.extract_exclusion_depths(scan_A2)
values_A2 = dilepton_limit.select_depths(scan_A2, all_depths_A2)

# Make a plot
x, y, z = clean_grid(scan_A2.mmed, scan_A2.mdm, values_A2)
make_plot(x, y, z, "A2", addText=None, addCurves=None, addPoints=True)

# And get another V2 by rescaling from this scan
rescaleA2 = Rescaler(scan_A2, all_depths_A2)
V2_limits = rescaleA2.rescale_by_br_leptons(target_gq=0.1,target_gdm=1,target_gl=0.01,model='vector')[(0.1,1.0,0.01)]
x, y, z = clean_grid(scan_A2.mmed, scan_A2.mdm, V2_limits)
make_plot(x, y, z, "V2_rescaled", addPoints = True)

# Now do a coupling-scan test.
# Very high resolution.
mMed_test = np.linspace(0,5000,501)
test_mass_scenarios = {
  "dmDecoupled" : [10000 for i in mMed_test],
  "dmLight" : [1.0 for i in mMed_test],
  "DPLike" : [i/3.0 for i in mMed_test]
}

test_coupling_scenarios = {
  "gl_lim" : {
    "test_gq" : [0.01, 0.1, 0.25],
    "test_gdm" : [0.0, 1.0],
    "test_gl" : np.logspace(np.log10(0.001),0,101),
  }
}

# For each of those, just make a 1D scan and get the limit depths.
# Then scale to couplings and go from there.
graph_dict = {}
for hypothesis, test_masses in test_mass_scenarios.items() :
  line_scan = DMAxialModelScan(
    mmed=mMed_test,
    mdm=test_masses,
    gq=0.1,
    gdm=1.0,
    gl=0.01
  )
  initial_depths = dilepton_limit.extract_exclusion_depths(line_scan)
  rescale_couplings = Rescaler(line_scan,initial_depths,"axial")

  for test_coupling, coupling_dict in test_coupling_scenarios.items() :
    test_gq = coupling_dict["test_gq"]
    test_gdm = coupling_dict["test_gdm"]
    test_gl = coupling_dict["test_gl"]

    # Loop through non-primary couplings and pull out plots
    if "gq_lim" in test_coupling :
      test_couplings = test_gq
      coupling_index = 0
      other_first, other_second = np.meshgrid(test_gdm, test_gl)
      others_tag = "gdm{0}_gl{1}"
    elif "gdm_lim" in test_coupling :
      test_couplings = test_gdm
      coupling_index = 1
      other_first, other_second = np.meshgrid(test_gq, test_gl)
      others_tag = "gq{0}_gl{1}"
    else :
      test_couplings = test_gl
      coupling_index = 2
      other_first, other_second = np.meshgrid(test_gq, test_gdm)
      others_tag = "gq{0}_gdm{1}"

    full_depths = rescale_couplings.rescale_by_br_leptons(coupling_dict["test_gq"],coupling_dict["test_gdm"],coupling_dict["test_gl"])
    limitplot_x, limitplot_y = np.meshgrid(mMed_test,test_couplings)    

    # Now we're looping over the other two couplings
    for other_one, other_two in zip(other_first.flatten(), other_second.flatten()) :
        these_depths = []

        for ci in test_couplings :
          full_couplings = [other_one, other_two]
          full_couplings.insert(coupling_index, ci)
          these_depths.append(full_depths[tuple(full_couplings)])

        # Plot with contour
        x, y, z = clean_grid(limitplot_x.flatten(), limitplot_y.flatten(), np.array(these_depths).flatten())
        thiskey = "axial_{0}_{1}_".format(hypothesis,test_coupling)+others_tag.format(other_one, other_two)
        drawContourPlotRough([[x, y, z]], addPoints = False, this_tag = thiskey+"_newdilep",plot_path = plot_path, xhigh=3000.,yhigh=0.5,vsCoupling=True)

        # Extract contour
        icontour = get_contours(x, y, z)[0]

        # Save TGraph
        igraph = ROOT.TGraph()
        for iicontour in icontour :
          for ix, iy in list(iicontour.exterior.coords) :
            igraph.AddPoint(ix,iy)
        graph_dict[thiskey] = igraph

outfile = ROOT.TFile.Open("test_dilepton_axial.root","RECREATE")
outfile.cd()
for key, graph in graph_dict.items() :
  graph.Write(key)
outfile.Close