import json
import csv

#Add complete path to PyCommitsWithDiffs.json file
file_path = "/PyCommitsWithDiffs.json"

# Function to read JSON from file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

output_csv = "output_extracted.csv"
print(file_path)
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
