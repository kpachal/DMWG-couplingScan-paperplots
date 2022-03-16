import os
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np

from shapely.geometry import Polygon as shapely_pol
from shapely.ops import unary_union
from shapely import validation
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.colors import ColorConverter

import colorsys

def get_aspect_ratio(ax) :
  ratio = 1.0
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  return abs((xright-xleft)/(ybottom-ytop))*ratio

def scale_lightness(matplotlib_col, scale_l):
    rgb = ColorConverter.to_rgb(matplotlib_col)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def get_contours(xvals, yvals, zvals) :

  # This separates into two sets of filled spaces:
  # spaces under 1 (that is, between 1 and 0)
  # and spaces over 1.
  cntr = plt.tricontourf(xvals, yvals, zvals, levels=[0,1.0,1e5], cmap="RdBu_r")

  # Extract them and organise them for returning.
  contours = {}

  # Loop through all polygons that have a specific intensity level
  for level,col in enumerate(cntr.collections) :

    # There can be multiple contours at each level
    sub_contours = []
    for contour_path in col.get_paths() :

        # Each path can also contain what should be
        # mulitple contours: split by MOVETO codes
        # where necessary
        vertices = contour_path.vertices
        codes = contour_path.codes
        idx = np.where(codes==Path.MOVETO)[0]
        vertex_segs = np.split(vertices,idx)[1:]

        # Now create and save polygons!
        for sub_segment in vertex_segs :
          new_shape = shapely_pol(sub_segment)

          # Check validity, fail if not valid
          if not new_shape.is_valid :
            print("Invalid polygon from extracted contour!")
            print(validation.explain_validity(new_shape))
            exit(1)

          sub_contours.append(new_shape)

    contours[level] = sub_contours
  
  return contours  

def merge_exclusions(contour_list) :

    # If there's only one, do nothing.
    if len(contour_list) < 2 :
      return contour_list[0]

    # No longer matters which input each contour is from:
    # flatten the input to a simple list of polygons
    contour_list = [item for sublist in contour_list for item in sublist]

    # Merge
    merged = unary_union(contour_list)

    # Split multipolygon back into list of polygons, if needed
    poly_list = []
    if merged.geom_type == 'MultiPolygon':
      poly_list = [poly for poly in merged.geoms]
    else :
      poly_list = [merged]

    return poly_list 

def drawContourPlotRough(grid_list, addPoints = False, this_tag = "default", plot_path = "plots", addText = "", xhigh=None, yhigh=None) :

  # Check output
  if not os.path.exists(plot_path) :
    os.makedirs(plot_path)

  # Object for plotting
  fig,ax=plt.subplots(1,1)

  levels = range(26)  # Levels must be increasing.
  usexhigh = xhigh if xhigh else 7500
  useyhigh = yhigh if yhigh else 1200
  ax.set_xlim(0, usexhigh)
  ax.set_ylim(0, useyhigh)
  plt.rc('font',size=17)
  ratio = get_aspect_ratio(ax)
  ax.set_aspect(ratio)

  ax.set_xlabel("m$_{ZA}$ [GeV]")
  ax.set_ylabel("m$_{\chi}$ [GeV]")   

  # Add text
  if addText :
    plt.figtext(0.2,0.75,addText,size=14)

  contour_list = []
  for grid in grid_list :
    cp = ax.tricontourf(grid[0], grid[1], grid[2], levels=levels, cmap='Blues_r')

    # Want points under contour, if adding them.
    if addPoints :
      # Separate into two populations: excluded and non excluded.
      xexcl,yexcl = [],[]
      xnon,ynon = [],[]
      for x,y,z in zip(grid[0], grid[1], grid[2]) :
        if z < 1. : 
          xexcl.append(x)
          yexcl.append(y)
        else :
          xnon.append(x)
          ynon.append(y)
      ax.scatter(xnon,ynon,color='red', marker='o',facecolors='none',linewidths=2)
      ax.scatter(xexcl,yexcl,color='white', marker='o',facecolors='none',linewidths=2)

    # Now add exclusion contour (if not doing official - harder to see with both)
    excontour = ax.tricontour(grid[0], grid[1], grid[2],levels=[1],colors=['r'],linewidths=[2])
    for path in excontour.collections[0].get_paths():
        this_contour = [path.vertices[:,0],path.vertices[:,1]]
        contour_list.append(this_contour)

  fig.colorbar(cp)

  #plt.savefig(plot_path+'/massmass_{0}.eps'.format(this_tag),bbox_inches='tight')
  plt.savefig(plot_path+'/massmass_{0}.pdf'.format(this_tag),bbox_inches='tight')

  plt.close(fig)

  return contour_list

def drawMassMassPlot(contour_groups, legend_lines, this_tag = "default", plot_path = "plots", addText = "",is_scaling=False, xhigh=None, yhigh=None) :

    # Check output
    if not os.path.exists(plot_path) :
        os.makedirs(plot_path)

    # Object for plotting
    fig,ax=plt.subplots(1,1)

    usexhigh = xhigh if xhigh else 7500
    useyhigh = yhigh if yhigh else 1200
    ax.set_xlim(0, usexhigh)
    ax.set_ylim(0, useyhigh)
    plt.rc('font',size=16)
    ratio = get_aspect_ratio(ax)
    ax.set_aspect(ratio)

    ax.set_xlabel("m$_{ZA}$ [GeV]")
    ax.set_ylabel("m$_{\chi}$ [GeV]")   

    if addText :
        if addText.count('\n') == 1 :
            plt.figtext(0.23,0.77,addText,size=14)
        elif addText.count('\n') == 2 :
            plt.figtext(0.23,0.72,addText,size=14)
        else : plt.figtext(0.23,0.82,addText,size=14)

    # Need 3 cute colours
    if is_scaling :
        ncols = len(contour_groups)
        fill_colours = [scale_lightness('cornflowerblue',0.5+i*1.0/ncols) for i in range(ncols)]
        line_colours = fill_colours
        line_width = 1
    else :
        colours_raw = ['cornflowerblue','turquoise','mediumorchid']
        fillOpacity = 0.5
        fill_colours = [ColorConverter.to_rgba(col, alpha=fillOpacity) for col in colours_raw]
        line_colours = colours_raw
        line_width = 2
    for contour_group,label_line,face_col,line_col in zip(contour_groups,legend_lines,fill_colours,line_colours) :
        for index, contour in enumerate(contour_group) :
            if index == 0 :
                patch = Polygon(list(contour.exterior.coords), facecolor=face_col, edgecolor=line_col, zorder=2, label=label_line,linewidth=line_width) #alpha=fillOpacity, 
            else :
                patch = Polygon(list(contour.exterior.coords), facecolor=face_col, edgecolor=line_col, zorder=2, label="_",linewidth=line_width) # alpha=fillOpacity,
            ax.add_patch(patch)
    ax.legend(fontsize=14)

    #plt.savefig(plot_path+'/massmass_{0}.eps'.format(this_tag),bbox_inches='tight')
    plt.savefig(plot_path+'/massmass_{0}.pdf'.format(this_tag),bbox_inches='tight')

    plt.close(fig)    

def drawDDPlot(contour_groups, legend_lines, this_tag = "default", plot_path = "plots", addText = "",ylabel="\sigma",is_scaling=False, transluscent=False, xhigh=None, ylow=None,yhigh=None) :

    # Check output
    if not os.path.exists(plot_path) :
        os.makedirs(plot_path)

    # Object for plotting
    fig,ax=plt.subplots(1,1)

    usexhigh = xhigh if xhigh else 2000
    useylow = ylow if ylow else 1e-46
    useyhigh = yhigh if yhigh else 1e-37
    ax.set_xlim(1, usexhigh)
    ax.set_ylim(useylow,useyhigh)
    plt.rc('font',size=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.rc('font',size=16)

    ax.set_xlabel("m$_{\chi}$ [GeV]", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    if addText :
        if addText.count('\n') == 1 :
            plt.figtext(0.16,0.15,addText,size=14)
        elif addText.count('\n') == 2 :
            plt.figtext(0.16,0.19,addText,size=14)
        else : plt.figtext(0.16,0.23,addText,size=14)

    # Need 3 cute colours
    if is_scaling :
        ncols = len(contour_groups)
        fill_colours = [scale_lightness('cornflowerblue',0.5+i*1.0/ncols) for i in range(ncols)]
        if transluscent :
          fill_colours = [ColorConverter.to_rgba(col, alpha=0.5) for col in fill_colours]
          line_colours = ['black' for i in fill_colours]
        else :
          line_colours = fill_colours          
        line_width = 1
    else :
        colours_raw = ['cornflowerblue','turquoise','mediumorchid']
        fillOpacity = 0.5
        fill_colours = [ColorConverter.to_rgba(col, alpha=fillOpacity) for col in colours_raw]
        line_colours = colours_raw
        line_width = 2
    for contour_group,label_line,face_col,line_col in zip(contour_groups,legend_lines,fill_colours,line_colours) :
        for index, contour in enumerate(contour_group) :
            if len(list(contour.exterior.coords)) == 0 : continue
            if index == 0 :
                patch = Polygon(list(contour.exterior.coords), facecolor=face_col, edgecolor=line_col, zorder=2, label=label_line,linewidth=line_width) 
            else :
                patch = Polygon(list(contour.exterior.coords), facecolor=face_col, edgecolor=line_col, zorder=2, label="_",linewidth=line_width)
            ax.add_patch(patch)
    ax.legend(fontsize=14)

    plt.savefig(plot_path+'/directdetection_{0}.pdf'.format(this_tag),bbox_inches='tight')

    plt.close(fig)    
            

