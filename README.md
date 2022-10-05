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

(Nvidia GPU ONLY)

`bash deepflow <folder of DICOM files> gpu`

**CPU Version**

`bash deepflow <folder of DICOM ZIP files> cpu`

**ARM Version**

*aarch64* is now supported. The latest Apple M1 (and later M2, M3 and so on) based macOS systems are now compatible with DeepFlow AI!

Tested on a MacBook Pro with M1 Max and 64 GB RAM

`bash deepflow <folder of DICOM ZIP files> arm`

*First run will pull the required Docker image which might take some time depending on the internet connection and the bandwidth.*

**Windows**

Make sure you have Windows 10 or higher with WSL (recommended) or Hyper-V configured Docker service.

**GPU**

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
ARM             1min 53s    2.79GB
```

```text
Host System
Windows & Linux
Alienware Area 51m R2
Windows 11 | Arch Linux
Kernel: 5.17.5-arch1-1
CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
GPU: NVIDIA RTX 2080 Super (notebook)
Docker version 20.10.14, build a224086349

macOS
MacBook Pro
M1 Max 10 cores
64 GB RAM
```

When using multiple GPUs or a powerful desktop/ server GPU, the speeds can go up drastically.

## Usage ##

A single ZIP file per sample is expected.
Results will be written in the `output` directory of the current folder.

## Description of scripts ##

**deepflow:** Caller script that pulls (if required) and runs the deepFlow Docker image with correct parameters and mounts directories

**deepflow_windows.bat:** Windows caller script that pulls (if required) and runs the deepFlow Docker image with correct parameters and mounts directories

**concat.sh:** Internal Docker VM script which compiles the results per run.

**deepFlow.py:** Main script that does all the calculations inside the Docker VM.

## Output ##

**CompiledReport<date:time>.tsv:** Tab-delimited report of all the samples from the given folder.

**<sample_name>.png:** Plot of the blood flow (mL/s) over the aortic valve in one heart cycle.

# Citation #

If you have used DeepFlow in any of your publications (thank you), don't forget to cite any one (or more) of the following papers:

```text
Genetic architecture of cardiac dynamic flow volumes for consideration in Nature Genetics (2022)
Bruna Gomes, Aditya Singha, Jack W O’Sullivana, David Amar, Mikhailo Kostur, Francois Haddad, Michael Salerno, Victoria N. Parikha, Benjamin Meder, Euan A. Ashley (Pre-print soon. Publication to follow.)
```
