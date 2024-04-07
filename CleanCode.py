import pandas as pd
import re
import numpy as np
import os

def clean_code(diff_text):
    cleaned_code = ''
    diff_lines = diff_text.split('\n')

    # Find the index where the code starts
    pattern = r'^^.*@@.*@@.*$'  # Regex pattern
    code_start_index = -1
    # Find the index of the first matching string
    matching_index = None
    for idx, string in enumerate(diff_lines):
      if re.match(pattern, string):
        code_start_index = idx
        last_at_index = string.rfind('@@')
        if last_at_index != -1:
          cleaned_code = string[last_at_index + 2:]
        break  # Stop searching once a match is found
    if code_start_index == -1:
      return ''
    code_start_index += 1

    # Extract the code lines
    code_lines = diff_lines[code_start_index:]

    # Exclude lines starting with '+' and '-'
    for line in code_lines:
        if line.startswith(('-')):
            cleaned_code += line[1:] + '\n'

    return cleaned_code.strip()


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming the CSV file is located in the same directory as the script
file_path = os.path.join(script_dir, 'output_extracted.csv')

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

codeSnippets = []
for index, row in data.iterrows():
  codeSnippets.append(clean_code(row['Diff']))
data['CodeSnippet'] = codeSnippets

# dropping the rows with empty code snippets
data['CodeSnippet'].replace('', np.nan, inplace=True)
data.dropna(subset=['CodeSnippet'], inplace=True)
data.to_csv('cleaned_code_snippets.csv', encoding='utf-8')

#New Data Path
file_path_new_data = os.path.join(script_dir, 'cleaned_code_snippets.csv')
new_data = pd.read_csv(file_path_new_data)

# Filtering/Cleaning
def filter_df(keyword, label):
  df = new_data[new_data['Keyword'].str.lower().str.contains(keyword)]
  df['label'] = label
  return df
df_sql = filter_df('sql', 0)
print(df_sql.shape)
df_xss = filter_df('xss', 1)
print(df_xss.shape)
df_xsrf = filter_df('xsrf', 2)
print(df_xsrf.shape)
df_open_red = filter_df('open redirect', 3)
print(df_open_red.shape)
df_command_injection = filter_df('command injection', 4)
print(df_command_injection.shape)
df_path_disclosure = filter_df('path disclosure', 5)
print(df_path_disclosure.shape)
df_remote_code_exe = filter_df('remote code execution', 6)
print(df_remote_code_exe.shape)

df_final = pd.concat([df_sql, df_xss, df_xsrf, df_open_red, df_command_injection, df_path_disclosure, df_remote_code_exe],axis=0,ignore_index=True)
df_final.head()
df_final.to_csv('final_dataset.csv', encoding='utf-8')
