# deepFlow - Docker #

**Purpose:** Outputs flow parameters from raw 2D (aorta) cardiac MRI

## Installation ##

**A. Clone the Repository**

`git clone https://github.com/Urban90/deepFlow.git`

**B. Run**

Requires Docker and if using GPU version, `NVIDIA Drivers` and `nvidia container utils`

`bash deepflow`

*First run will pull the required Docker image which might take some time depending on the internet connection and the bandwidth.*

## Usage ##

A GUI will ask you to select the folder containing zipped DICOM files.
A single ZIP file per sample is expected.
Results will be written in the `output` directory of the current folder.

***Docker mounts your `HOME` directory inside `data`***
