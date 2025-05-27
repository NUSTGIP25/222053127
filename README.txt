Project Title: RERA Site Suitability Viewer by Jeffrey N.S. Shigwedha (222053127)

This application helps visualize and analyze geospatial data for renewable energy site suitability using AHP (Analytic Hierarchy Process).

Before running the application, ensure you have Python 3.9 or higher installed.

Installation Instructions:

1. Clone or download the project folder to your local machine and place all the files in a folder named "rera-python", this will be your working directory.

2. Open a terminal or command prompt in the working directory.

3. (Optional) Create a virtual environment:

	python -m venv venv

4. (Optional) Activate the virtual environment:
	
	venv\Scripts\activate

5. Install required libraries using the requirements textfile:

	pip install -r requirements.txt

6. Download the datasets (a folder called "data") needed for this project by using this link, place it in the "rera-python" once downloaded:

	https://mega.nz/folder/eJ8GVaxS#ROYxmGcmJhjQjJOKI-8s8g 

7. Ensure the working directory folder is structured as so:
	|- rera-python
		|- .ipynb_checkpoints
		|- __pycache__
		|- data
			|- DNI.tif 
			|- Land Tenure.geojson 
			|- Major Rivers.geojson 
			|- NAM_wind-speed_100m.tif 
			|- NamRoads.geojson 
			|- NamDEM.tif 
			|- Perennial Catchment Areas.geojson
			|- Powerlines.geojson 
			|- R.E. Sources.geojson 
			|- Substations.geojson 
		|- myenv (optional)
		|- temp
		|- requirements.txt
		|- README.txt
		|- map.html
		|- RERA_DataViewer.py
		|- AHP.py
		|- temp_overlay.png

8. Running the App:

	1. To start the main map viewer, run this in your terminal:
		python RERA_DataViewer.py
	
	2. To run the AHP suitability analysis, run this in your terminal:
		python AHP.py

