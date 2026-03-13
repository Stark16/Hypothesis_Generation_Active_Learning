import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import json
import sys
import pandas as pd
sys.path.append("/home/ppathak2/Hypothesis_Generation_Active_Learning")
from tqdm import tqdm

import MAIN_context_tree_emb_pipeline

def read_bats_data(PATH_BATS_dataset_root):
    BATS_categories = {}
    for category in os.listdir(PATH_BATS_dataset_root):
        path_category = os.path.join(PATH_BATS_dataset_root, category)
        if os.path.isdir(path_category):
            category_content = [] 

            for category_type_file in tqdm(os.listdir(path_category), desc=f'Processing - {category}'):
                path_category_type_file = os.path.join(path_category, category_type_file)
                id, relation_category = category_type_file.rsplit('].')[0].split(' [')
                with open(path_category_type_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split()
                        row = {"Id" : id, "relation-category": relation_category, 
                        "value1": line[0], "value2": line[1]}
                        category_content.append(row)
            BATS_categories[category[2:]] = category_content
    return BATS_categories

def clean_keyword(keyword):
    if '/' in keyword:
        k1 = []
        keyword = keyword.split('/')
        for kw in keyword:
            k1.append(kw.strip())
        keyword = k1
    if isinstance(keyword, str):
        keyword = [keyword.strip()]
    return keyword

def preprocess_keywords(keyword1, keyword2):
    keyword1 = clean_keyword(keyword1)
    keyword2 = clean_keyword(keyword2)
    keywords = []
    keywords.extend(keyword1)
    keywords.extend(keyword2)
    return keywords

def run_test_on_df(data_frame_name:str, df: pd.DataFrame, PATH_output_dir: str):
    for index, row in tqdm(df.iterrows(), desc=f'Running tests for {data_frame_name}'):
        keyword1 = row['value1']
        keyword2 = row['value2']
        domain = row['relation-category']

        keywords = preprocess_keywords(keyword1, keyword2)

        MAIN_context_tree_emb_pipeline.run(PATH_output_dir, domain, keywords, temperature=40, num_trees=1, depth_cap=3)

def run_tests():
    PATH_BATS_dataset_root = "/home/ppathak2/Hypothesis_Generation_Active_Learning/datasets/BATS_3.0"
    BATS_data = read_bats_data(PATH_BATS_dataset_root)

    # Converting the data to dataframe for easier handling
    DF_Inflectional_morphology = pd.DataFrame(BATS_data['Inflectional_morphology'])
    DF_Derivational_morphology = pd.DataFrame(BATS_data['Derivational_morphology'])
    DF_Encyclopedic_semantics = pd.DataFrame(BATS_data['Encyclopedic_semantics'])
    DF_Lexicographic_semantics = pd.DataFrame(BATS_data['Lexicographic_semantics'])

    # Now, I will make this whole thing into two batches, first is the Encyclopedic/[male - female] and second is the rest
    DF_Batch1 = DF_Encyclopedic_semantics.where(DF_Encyclopedic_semantics['relation-category'] ==  'male - female').dropna()
    # And removing those rows from the original dataframe
    DF_Encyclopedic_semantics = DF_Encyclopedic_semantics.drop(DF_Batch1.index)

    # Putting everything in a list for easier iteration
    test_batches = [['female-male-batch', DF_Batch1], 
                    ['Inflectional_morphology', DF_Inflectional_morphology],
                    ['Derivational_morphology', DF_Derivational_morphology], 
                    ['Lexicographic_semantics', DF_Lexicographic_semantics], 
                    ['Encyclopedic_semantics', DF_Encyclopedic_semantics]]
    
    for test_batch in test_batches:
        batch_name = test_batch[0]
        df = test_batch[1]
        PATH_output_dir = os.path.join("/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/BATS_BERT", batch_name)
        run_test_on_df(batch_name, df, PATH_output_dir)

if __name__ == "__main__":
    run_tests()