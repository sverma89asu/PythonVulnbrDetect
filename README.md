# PythonVulnbrDetect
This project is aimed to detect different software vulnerabilities in python source code using machine learning 

# Requirements: 
```
pip install -r requirements.txt
```
### Running the code to train the models
Dataset:
Download the dataset from https://zenodo.org/records/3559203#.XeRoytVG2Hs. This is a large dataset.


Download the PyCommitsWithDiffs.json 


Run :-
```
python JsonToCsvConverter.py
```
It will ask you to select 
Select the downloaded PyCommitsWithDiffs.json and click Open.
This will extract the data to csv and save in output_extracted.csv .


Run :-
```
python CleanCode.py
```


This will generate cleaned_code_snippets.csv and final_dataset.csv


Run :-
```
python MLPredictions.py
```
This will display results in the console and generate images.
Close the image to let the program proceed.

