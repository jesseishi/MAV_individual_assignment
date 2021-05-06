# Gate detection using feature clustering
#### Individual assignment for AE4317 Autonomous Flight of Micro Air Vehicles
#### By: Jesse Hummel

The goal of this assignment was to develop a gate detection algorithm and write an article about this. Both the
assignment and final article are present in this repo as .pdf files.

In short the goal is to estimate the location of the corners of the gate from an image. The images were provided by the
course, but not present in this repo. The code assumes that the images are located in a folder called WashingtonOBRace
that is located in the same folder as this repo. If it is located elsewhere on your computer, search for `data_folder`
in the code and replace it with the appropriate path.

The structure of this repo is as follows:
* src/
    * final/: Contains the source code of the final algorithm. In `GateDetector.py`, the GateDetector class is defined,
      which contains the whole algorithm. In `TestGateDetector.py` a class is made that can check if the corner coordinates
      are within the specified maximum error.
    * prototyping/: Contains all prototyping efforts into different types of algorithms before the decision was made to
      do cluster based detection.
* test/: Contains all the code used to run tests on the GateDetector.
    * `detect_one_image.py`: Used to run the algorithm with a single setting on a single image. This can also visualize
      the internal workings of the algorithm.
    * `detect_batch.py`: Used to do batch runs where algorithm parameters may be varied. The results are stored in the
      results folder (which is in the .gitignore).
    * `find_good_settings.py`: After a batch run, this script may be used to make heatmaps and find combinations of
       parameters where the algorithm performs well.
    * `distance_correlation.py`: After a batch run, this script may be used to investigate the dependency of gate
       size (=gate distance) to detection rate.
    * `make_ROC_data.py`: Is used to generate the data to make the ROC curves.
    * `make_ROC_curve.py`: After having generated data, this script can be used to make the ROC plot.
