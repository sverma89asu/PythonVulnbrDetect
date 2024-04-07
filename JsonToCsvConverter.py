import tkinter as tk
from tkinter import filedialog
import json
import csv

# Create a Tkinter root widget
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to select a file
file_path = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON files", "*.json")])

# Check if a file was selected
if file_path:
    print("Selected file:", file_path)
    # Your further processing code here
else:
    print("No file selected.")
    

# Function to read JSON from file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Path to the JSON file
#file_path = r'C:\Users\virub\Downloads\plain_command_injection [MConverter.eu].json'
output_csv = "output_extracted.csv"

# Read JSON data from file
json_data = read_json_file(file_path)

#i = 0
# Function to write commit data to a CSV file
def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='',encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Repository', 'Commit Hash', 'URL', 'HTML URL', 'SHA', 'Keyword', 'Diff'])
        for repository, commits in data.items():
           # checker = 0
            for commit_hash, commit_details in commits.items():
                if '.py' in commit_details['diff'][:100]:
                    writer.writerow([repository, commit_hash, commit_details['url'], commit_details['html_url'], 
                                     commit_details['sha'], commit_details['keyword'], commit_details['diff']])
                    

# Write commit data to CSV file
write_to_csv(output_csv, json_data)
