# KDS_AdvancedMachineLearning_project

Repository containing source code and results of final mini-project for Advanced Machine Learning, course of Data Science master at ITU.
Team members: Thomas Fosdam Claudinger, Emil Haldan, Jorge del Pozo Lerida

### Prequisites
Install packages described in requirements.txt

# Task: Model how the brain represents visual information
{Central problem, domain}

## Goals
* Do a comprehensive pre-processing and some visualisation of the fMRI data. Can you see patterns in the data for either input category?
* Train a voxel-wise encoding model (deep CNN network) to predict fMRI responses in different brain regions from the stimulus data as input. You might consider vanilla versus pre-trained image processing networks. Can you identify an architecture (and meta-parameter settings) that predicts well and remains bio-inspired regarding the hierarchy?
* Compare predictions of the individual layers (model) with activations of different regions (imaging data), e.g. through heatmaps. How does the cortex hierarchy to the model's hierarchy? Do you observe any patterns?
### Secondary goals
* Can you suggest which brain regions preserve spatial information? Apply image transformations randomly before feeding to the model and observe the change in encoding accuracy.
* Can you identify in which regions the brain might encode categorical information? For this, you might compare representation similarities (e.g. via RSA).

# The dataset
{data characteristics}

# Methods
{Central method: chosen architecture and training mechanisms, with a brief justification if non-standard}

# Experiments & results
{present and explain results, e.g. in simple accuracy tables over error graphs up to visualisations of representations and/or edge cases – keep it crisp}

# Discussion
{summarise the most important results and lessons learned (what is good, what can be improved)}