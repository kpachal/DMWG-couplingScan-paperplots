import pickle

from basic_plotter import *

test_gq = [0.25,0.15,0.1,0.05]
test_gdm = [1.0,0.75,0.5,0.25]
test_gl = [0.1,0.05,0.01,0.0]

plotlims = (3500,1700)

# Load pickle files with polygons
for model in ['vector','axial'] :
    with open('{0}_exclusion_contours.pkl'.format(model), "rb") as poly_file:
        loaded_polygons = pickle.load(poly_file)
            
        # Grid of plots: 
        for gdm in test_gdm :
            for gq in test_gq :
                contours_list_couplingscan = []
                legend_lines_couplingscan = []
                for gl in test_gl :
                    contours_list = []
                    legend_lines = []
                    for signature in ['dijet','monophoton'] :
                        exclusions = loaded_polygons[signature][(gq, gdm, gl)]
                        contours_list.append(exclusions)
                        legend_lines.append(signature)
                    # First set of plots: 3 contours, one plot for every coupling combo
                    label_line =  "{0}, g$_{5}$={2}\ng$_{4}$={1}, g$_{6}$={3}".format(("Axial-vector" if 'axial' in model else "Vector"),gq,gdm,gl,"q","\chi","l")
                    drawMassMassPlot(contours_list, legend_lines, this_tag = model+"_gq{0}_gdm{1}_gl{2}".format(gq, gdm, gl), plot_path = "plots/massmass/", addText=label_line, xhigh=plotlims[0], yhigh=plotlims[1])
                    if all(not i for i in contours_list) : continue
                    full_polygons = merge_exclusions(contours_list)
                    contours_list_couplingscan.append(full_polygons)
                    legend_lines_couplingscan.append("g$_{0}$={1}".format("l",gl))
                # Second set of plots: merge all contours; fix gq and gdm and vary gl.
                # Note this is not meaningful where we don't have dilepton projections - skip then.
                label_line = "{0}\ng$_{3}$={1}, g$_{4}$={2}".format(("Axial-vector" if 'axial' in model else "Vector"),gq,gdm,"q","\chi")
                drawMassMassPlot(contours_list_couplingscan,legend_lines_couplingscan, this_tag = model+"_gq{0}_gdm{1}".format(gq,gdm), plot_path = "plots/massmass/", addText = label_line,is_scaling=True, xhigh=plotlims[0], yhigh=plotlims[1])
            # Need second set of plots with gl and gdm fixed instead:
            for gl in test_gl :
                contours_list_couplingscan = []
                legend_lines_couplingscan = []
                for gq in test_gq :
                    contours_list = []
                    for signature in ['dijet','monophoton'] :
                        exclusions = loaded_polygons[signature][(gq, gdm, gl)]
                        contours_list.append(exclusions)
                    if all(not i for i in contours_list) : continue
                    full_polygons = merge_exclusions(contours_list)
                    contours_list_couplingscan.append(full_polygons)
                    legend_lines_couplingscan.append("g$_{0}$={1}".format("q",gq))
                label_line = "{0}\ng$_{3}$={1}, g$_{4}$={2}".format(("Axial-vector" if 'axial' in model else "Vector"),gdm,gl,"\chi","l")
                drawMassMassPlot(contours_list_couplingscan,legend_lines_couplingscan, this_tag = model+"_gl{0}_gdm{1}".format(gl,gdm), plot_path = "plots/massmass/", addText = label_line,is_scaling=True, xhigh=plotlims[0], yhigh=plotlims[1])
        # And now third set of plots with gq and gl fixed:
        for gq in test_gq :
            for gl in test_gl :
                contours_list_couplingscan = []
                legend_lines_couplingscan = []
                for gdm in test_gdm :
                    contours_list = []
                    legend_lines = []                    
                    for signature in ['dijet','monophoton'] :
                        exclusions = loaded_polygons[signature][(gq, gdm, gl)]
                        contours_list.append(exclusions)
                    if all(not i for i in contours_list) : continue
                    full_polygons = merge_exclusions(contours_list)
                    contours_list_couplingscan.append(full_polygons)
                    legend_lines_couplingscan.append("g$_{0}$={1}".format("\chi",gdm))
                label_line = "{0}\ng$_{3}$={1}, g$_{4}$={2}".format(("Axial-vector" if 'axial' in model else "Vector"),gq,gl,"q","l")
                drawMassMassPlot(contours_list_couplingscan,legend_lines_couplingscan, this_tag = model+"_gq{0}_gl{1}".format(gq,gl), plot_path = "plots/massmass/", addText = label_line,is_scaling=True, xhigh=plotlims[0], yhigh=plotlims[1])
