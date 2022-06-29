# This is the official repository for: Champagne Taste on a Beer Budget: Better Budget Utilisation in Multi-label Adversarial Attacks

## Run instructions:
* Use requirements.txt to isntall the required packages
* In the .json config files of the model(s) (example: asl_coco.json) change the resume/model_path variable to the location ()of the model paramters (example: ../../models/tresnetxl-asl-voc-epoch80)
* Make sure there is a folder called experiment_results in the repo root directory
* Then enter the following command: python script_name.py classifier dataset path_to_dataset
	* example: python attack.py asl_nuswide ../../NUS_WIDE NUS_WIDE
	* For other options such as batch_size, threshold, etc, run without arguements for help
	* Results are stored as Numpy files and can be printed with scripts in plots folder