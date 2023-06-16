# Cell signal, and cell volume analysis (Sfp1)

## Contents of this repository
This repository contains multiple directories that apply a similar sort of analysis on microscopy imaging data.  
All three sub-folders contain a Jupyter notebook, as well as a script. The notebook was/is used for exploratory analysis, 
while the script is an efficient way to execute the complete analysis.

### Sub-folder [gfp-only-analysis](sfp1-gfp-analysis)
Provides insight into the Sfp1 difference in nuclear and cytoplasmic signal intensity. In this particular case, the only data
used from the tif images is the GFP channel, which tracks Sfp1. Local/adaptive thresholding is performed to find the nucleus
its centroid. This is used to calculate the average nuclear signal. The cytoplasmic signal is taken as an average of the rest
of the cell (so omitting the nucleus).

### Sub-folder [rfp-scarlet-analysis](sfp1-mScarlet-analysis)
Provides insight into the nuclear and cytoplasmic signal intensity. In this particular case, the only data used from the 
tif images is the RFP channel. Local/adaptive thresholding is performed to find the nucleus its centroid.  This is used to
calculate the average nuclear signal. The cytoplasmic signal is taken as an average of the rest
of the cell (so omitting the nucleus). Since these images diverge from the other ones, the settings and approach for the
thresholding are/is different. 

### Sub-folder [volume-analysis](volume-analysis)
In this case, instead of calculating the nuclear to cytoplasmic signal strengths, the nuclear and whole-cell volumes are 
calculated from through whole-cell masking and nuclear masking. This is also done using local/adaptive thresholding. Outliers
are removed using the IQR-approach and the data is stored in a dataframe.

## Setup
### Environment
1. Make sure Python is installed (preferably version 3.9)
2. Create a virtual environment like this: ```python3 -m venv preferred-venv-location```.
3. Activate the environment you just created by running ```source path-to-your-venv/bin/activate``` or ```path-to-your-venv/Scripts/activate``` on Windows.
4. Install all necessary dependencies by running: ```pip3 install -r requirements.txt```.

### General instructions; script execution
1. Make sure to check all hard-coded filepaths at the top of the main file(s) are configured correctly.
2. Make sure the data is in the correct place.
3. Run the script using python.
4. Debug any minor error messages (such as incorrect paths etc.).
5. Rerun.
