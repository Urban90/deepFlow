@echo off
SET input=%1
SET cdir=%~dp0
echo Welcome to DeepFlow!
if %2%==cpu (
	echo CPU mode selected. Performance might be slower.
	docker run --rm -v "%cdir%output:/output" -v "%cdir%assets:/assets" -v %input%:/input saditya88/deepflow:cpu /input
	) else if %2%==gpu (
	echo Using GPU acceleration. Make sure that the NVIDIA Drivers and container toolkits are installed.
	docker run --rm --gpus all -v "%cdir%output:/output" -v "%cdir%assets:/assets" -v %input%:/input saditya88/deepflow:gpu /input
	) else (echo Unknown parameter specified: %2%)
echo "All done."
echo "Thanks for using deepFlow"