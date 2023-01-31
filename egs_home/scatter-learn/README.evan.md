In the evan_commits directory, there's are codes to automate the scatter-learn 
simulation. 

They are:
 - main.sh
   A single call code, which takes in the "pattern_ind" and the "ncase" for the
   simulation. "pattern_ind" selects one of several scatter-mask, contained in
   generate-scatter-masks.py. "ncase" is a input variable used in the .egsinp file

 - generate-scatter-masks.py
   Python code, generates a the include-scatter-media...dat file for a simulation
   instance. The include...dat file is generated in a way that the material choices
   for each voxel, together, create a 3D pattern.
   These patterns are:
     > Square slabs stacked in x
     > Square slabs stacked in y
     > Square slabs stacked in z
     > Checker-board of coloumns in x
     > Checker-board of coloumns in y
     > Checker-board of coloumns in z
     > 3D Checker-board

 - scatter_mask_functions.py
   Python code which defines the patterns used in generate_scatter-masks.py. 
  
 - scatter-learn-placeholder.egsinp
   A copy of scatter-learn.egsinp, but with a dummy placeholder value for the
   "ncase" and the scattering object's "includefile". It is copied then 
   stream-edited for each simulation instance.

 - output_parse_-_detector.sh
   Shell script which parses the output of scatter-learn...egslog into a 4xN .csv
   file. Must be normalized prior to regression

 - output_parse_-_scatter_media.sh
   Shell script wich parases the media definition file for the scattering object into 
   a 4xN .csv file. Must be normalized prior to regression.

Other codes not included in automation:
 - ParallelSweep.sh
   Shell script, not completed, for the automation of dataset generation. 

 - scatter-mask-python-demo.py
   Python code, which plots the 7 different masks available presentl/

 - README.evan.md
   This document.
   

