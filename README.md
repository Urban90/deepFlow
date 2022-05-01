# deepFlow - Docker #

**Purpose:** Outputs flow parameters from raw 2D (aorta) cardiac MRI

## Installation ##

### Requirements ###

```text
Docker
NVIDIA Driver (GPU version)
NVIDIA Docker Container utilities (GPU version)
```

**Clone the Repository**

`git clone https://github.com/Urban90/deepFlow.git`

### Run ###

Change directory into the newly cloned DeepFlow folder

`cd deepflow`

**Linux/ Unix and macOS**

**GPU Version** (default)

`bash deepflow <folder of DICOM files> gpu`

**CPU Version**

`bash deepflow <folder of DICOM ZIP files> cpu`

*First run will pull the required Docker image which might take some time depending on the internet connection and the bandwidth.*

**Windows**

Make sure you have Windows 10 or higher with WSL (recommended) or Hyper-V configured Docker service.

**GPU**

*In our experience, the GPU version is a bit buggy, not due to the code but because of some WSL2/ CUDA and Docker issues*

`deepflow_windows.bat <folder of DICOM ZIP files> gpu`


**CPU**

`deepflow_windows.bat <folder of DICOM ZIP files> cpu`

*Logs are not written to the disk in the Windows version*

## Benchmarking ##

In our tests, the CPU and GPU versions didn't have much of a difference in overall time but your mileage might vary.

We selected ten random samples from the UK Biobank repository and perfomed the test on the host system in triplicate.

```text
Image version   Mean Time   Uncompressed size
GPU             2min 37s    6.71 GB
CPU             2min 31s    2.34 GB
```

```text
Host System
Alienware Area 51m R2
Arch Linux
Kernel: 5.17.5-arch1-1
CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
GPU: NVIDIA RTX 2080 Super (notebook)
Storage: SanDisk SSD SD9SN8W512G USB-C
Docker version 20.10.14, build a224086349
```

When using multiple GPUs or a powerful desktop/ server GPU, the speeds can go up drastically.

## Usage ##

A single ZIP file per sample is expected.
Results will be written in the `output` directory of the current folder.

## Description of scripts ##

**deepflow:** Caller script that pulls (if required) and runs the deepFlow Docker image with correct parameters and mounts directories

**concat.sh:** Internal Docker VM script which compiles the results per run.

**deepFlow.py:** Main script that does all the calculations inside the Docker VM.

## Output ##

**CompiledReport<date:time>.tsv:** Tab-delimited report of all the samples from the given folder.

**<sample_name>.png:** Plot of the blood flow (mL/s) over the aortic valve in one heart cycle.

# Citation #

If you have used DeepFlow in any of your publications (thank you), don't forget to cite any one (or more) of the following papers:

```text
Natural history and GWAS of aortic valve regurgitation - a UK Biobank study (2022)
Gomes, B. Singh, A., O'Sullivan, J, Ashley, E. (In Writing)
```
