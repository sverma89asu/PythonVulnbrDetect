# CSE 565 Assignment 2 Group 8
# Members
1. Shikha Verma
2. Viresh Vilas Bhurke
3. Akshat Jain
4. Yeain Shukla

# PythonVulnbrDetect
This project is aimed to detect different software vulnerabilities in python source code using machine learning 

All the generated csv files are uploaded on the drive and can be used to run the MLPredictions.py directly instead of running the other three python file. https://drive.google.com/drive/folders/1wdTOo2Fj9za1IiIfLJi9-kc4eYMSLFDS

# Create a New Virtual Env: 
```
python3 -m venv cse
source cse/bin/activate
```
# Requirements: 
```
pip install -r requirements.txt
```
### Running the code to train the models
Dataset:
Download the dataset from https://zenodo.org/records/3559203#.XeRoytVG2Hs. This is a large dataset.


Download the PyCommitsWithDiffs.json 


Run :-

Before running the .py file make sure to set the filepath correctly.
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

