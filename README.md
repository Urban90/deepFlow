# deepFlow - Docker #

**Purpose:** Outputs flow parameters from raw 2D (aorta) cardiac MRI

## Installation ##

### Requirements ###

```text
Docker
NVIDIA Driver (GPU version)
NVIDIA Docker Container utilities (GPU version)
```

**A. Clone the Repository**

`git clone https://github.com/Urban90/deepFlow.git`

**B. Run**

Change directory into the newly cloned DeepFlow folder

`cd deepflow`

**GPU Version** (default)

`bash deepflow <folder of DICOM files>`

**CPU Version**

`bash deepflow <folder of DICOM ZIP files> cpu`

*First run will pull the required Docker image which might take some time depending on the internet connection and the bandwidth.*
s
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

When using multiple GPUs or a power desktop/ server GPU, the speeds can go up drastically.

## Usage ##

A single ZIP file per sample is expected.
Results will be written in the `output` directory of the current folder.

***Docker mounts your `HOME` directory inside `data`***

## Description of scripts ##

**deepflow:** Caller script that pulls (if required) and runs the deepFlow Docker image with correct parameters and mounts directories

**concat.sh:** Internal Docker VM script which compiles the results per run.

**deepFlow.py:** Main script that does all the calculations inside the Docker VM.
