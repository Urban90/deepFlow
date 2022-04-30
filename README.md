# deepFlow - Docker #

**Purpose:** Outputs flow parameters from raw 2D (aorta) cardiac MRI

## Installation ##

### Requirements ###

```text
Docker
X Window manager (X11/ XORG etc.)
NVIDIA Driver (GPU version)
NVIDIA Docker Container utilities (GPU version)
```

**A. Clone the Repository**

`git clone https://github.com/Urban90/deepFlow.git`

**B. Run**

**GPU Version** (default and highly recommended)

`bash deepflow`

**CPU Version**

`bash deepflow cpu`

*First run will pull the required Docker image which might take some time depending on the internet connection and the bandwidth.*

## Usage ##

A GUI will ask you to select the folder containing zipped DICOM files.
A single ZIP file per sample is expected.
Results will be written in the `output` directory of the current folder.

***Docker mounts your `HOME` directory inside `data`***

## Description of scripts ##

**deepflow:** Caller script that pulls (if required) and runs the deepFlow Docker image with correct parameters and mounts directories

**concat:** Internal Docker VM script which compiles the results per run.

**deepFlow.py:** Main script that does all the calculations inside the Docker VM.

