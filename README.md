# Nup133 cell-cycle volume analysis

## Environment setup

1. Make sure Python is installed (preferably a version > 3.9)
2. Create a virtual environment like this: ```python3 -m venv preferred-venv-location```
3. Activate the environment you just created by running ```source path-to-your-venv/bin/activate```
4. Install all necessary dependencies by running: ```pip3 install -r requirements.txt```

## What does this repository contain?
### Jupyter Analysis file (volume_analysis.ipynb)
The ```volume_analyis.ipynb``` file was created with the purpose of figuring out what was the best way of going about
the nuclear segmentation of yeast cells. This is by no means an efficient script to perform the analysis, but rather a file
that shows how the final results were obtained.

#### Setup
1. Make sure to check all hard-coded filepaths in the first code chunk are configured correctly.
2. When running it for the first time, make sure the data is in the correct place.
3. Run the notebook file chunk by chunk when you're uncertain.

### Volume analysis script
The ```volume_analysis.py``` file was created after ```volume_analyis.ipynb```, and is the optimized version of
the entire workflow. It thus only contains the parts that are essential, and which contribute to the most optimal result.
#### Setup
1. Make sure to check all hard-coded filepaths at the top of the file are configured correctly.
2. Run the script using python
