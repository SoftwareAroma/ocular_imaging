import pandas as pd
import kaggle
import os

kaggle.api.authenticate()
# kaggle.api.print_config_values()
# print('Download directory:', kaggle.api.get_default_download_dir())

# Initialize variables to hold all datasets
all_fundus_datasets = []

# Loop to get all pages of datasets
page = 1
while True:
    kaggle_fundus_datasets = kaggle.api.dataset_list(search='fundus', page=page)
    # print(f'Page {page}: {len(kaggle_fundus_datasets)} datasets')
    ### If no datasets are returned, break the loop
    if not kaggle_fundus_datasets:
        break
    ### Add the datasets to the list
    all_fundus_datasets.extend(kaggle_fundus_datasets)
    ### Move to the next page
    page += 1

### Verify the total number of datasets
print(f'Total Datasets Found: {len(all_fundus_datasets)}')

### Write the datasets to a file for inspection
with open('kaggle_fundus_datasets.txt', 'w') as f:
    for ds in all_fundus_datasets:
        f.write(f"{ds.ref}\n")

### Read the dataset references into a DataFrame
df = pd.read_csv("kaggle_fundus_datasets.txt", sep="/", header=None, names=["user_id", "dataset_name"])

### Directory to save the datasets
download_path = '/mnt/data/RiskIntel/ocular_imaging/datasets'

### Loop through each dataset reference and download it if not already downloaded
for index, row in df.iterrows():
    dataset_ref = row['user_id'] + '/' + row['dataset_name']
    dataset_path = os.path.join(download_path, row['dataset_name'])

    ### Check if the dataset directory already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already downloaded: {dataset_ref}")
        continue
    
    print(f"Downloading dataset: {dataset_ref}")
    kaggle.api.dataset_download_files(
        dataset_ref, 
        path=download_path, 
        unzip=True
    )

