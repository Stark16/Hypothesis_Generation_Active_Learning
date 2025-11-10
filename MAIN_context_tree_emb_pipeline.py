## This is the main file to run the Hierarchical Context Tree Embedding Pipeline

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from hierarchical_embedding_pipeline import HierarchicalEmbPipeline

def plot_time_analysis(TIME_context_tree, TIME_embedding_tree,  TIME_total: float, PATH_output_dir:str, node_per_keyword: list):
    """ 
    Function that plots the TIME_context_tree and TIME_embedding_tree as an interactive line plot using Plotly Express.
    The plot includes hover tooltips showing the number of LLM calls for the Context Tree Creation Time.
    The final plot is saved as a standalone HTML file.
    """
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Keyword Index': range(len(TIME_context_tree)),
        'Context Tree Creation Time': TIME_context_tree,
        'Hierarchical Embedding Time': TIME_embedding_tree,
        'LLM Calls': node_per_keyword
    })

    # Melt the DataFrame to have a long format for easier plotting of multiple lines
    df_melted = df.melt(id_vars=['Keyword Index', 'LLM Calls'], 
                        value_vars=['Context Tree Creation Time', 'Hierarchical Embedding Time'],
                        var_name='Trace', value_name='Time (seconds)')

    # Create the line plot
    fig = px.line(df_melted, x='Keyword Index', y='Time (seconds)', color='Trace',
                  title=f'Time Analysis of Hierarchical Embedding Pipeline<br>Total Time: {TIME_total:.2f}s',
                  markers=True,
                  color_discrete_map={
                      'Context Tree Creation Time': 'orange',
                      'Hierarchical Embedding Time': 'blue'
                  })

    # Customize hover data for the 'Context Tree Creation Time' trace
    fig.for_each_trace(
        lambda trace: trace.update(customdata=df['LLM Calls'],
                                   hovertemplate="<b>Keyword Index</b>: %{x}<br>" +
                                                 "<b>Time (seconds)</b>: %{y:.2f}s<br>" +
                                                 "<b>LLM Calls</b>: %{customdata}<extra></extra>")
        if trace.name == 'Context Tree Creation Time' else trace.update(
            hovertemplate="<b>Keyword Index</b>: %{x}<br>" +
                          "<b>Time (seconds)</b>: %{y:.2f}s<extra></extra>"
        )
    )

    # Calculate and add average lines
    avg_context_time = df['Context Tree Creation Time'].mean()
    avg_embedding_time = df['Hierarchical Embedding Time'].mean()

    fig.add_hline(y=avg_context_time, line_dash="dash", line_color="orange",
                  annotation_text=f'Avg Context Tree Time: {avg_context_time:.2f}s', 
                  annotation_position="bottom right")
    fig.add_hline(y=avg_embedding_time, line_dash="dash", line_color="blue",
                  annotation_text=f'Avg Embedding Time: {avg_embedding_time:.2f}s', 
                  annotation_position="top right")

    # Update layout
    fig.update_layout(
        xaxis_title='Keyword Index',
        yaxis_title='Time (seconds)',
        legend_title_text='Trace'
    )

    # Save to HTML
    os.makedirs(PATH_output_dir, exist_ok=True)
    PATH_plot = os.path.join(PATH_output_dir, 'interactive_time_analysis_plot.html')
    fig.write_html(PATH_plot, auto_open=False)


def run(PATH_output_dir, domain, keywords, temperature=40, num_trees=1, depth_cap=3,
        batch_query=True, no_history=True, log_time: bool=True):

    TIME_code_start = time.time()
    TIME_context_tree = []
    TIME_embedding_tree = []
    node_per_keyword_list = []

    for keyword in keywords:
        keyword = keyword.strip()
        os.makedirs(PATH_output_dir + '/' + domain, exist_ok=True)
        print("\n\t\t", "-"*50, " <", keyword, "> From Domain - <", domain, "> ", "-"*50, "\n")
        if keyword in (os.listdir(PATH_output_dir + '/' + domain)):
            print(f"\n\t\t Skipping {keyword} as it already exists in the output directory.")
            continue

        OBJ_HierarchEmbPipe = HierarchicalEmbPipeline(keyword, domain, BATCH_QUERY=True)
        OBJ_HierarchEmbPipe.OBJContextTree.PATH_output_trees = PATH_output_dir
        
        # Setting up some configurable arguments-
        OBJ_HierarchEmbPipe.OBJContextTree.generation_args['temperature'] = temperature/100
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["batch_query"] = batch_query
        OBJ_HierarchEmbPipe.MODEL_ARGS_gen_llm["no_history"] = no_history

        t_ct, t_et, node_per_keyword = OBJ_HierarchEmbPipe.create_embedding(num_trees, depth_cap)

        TIME_context_tree.append(t_ct)
        TIME_embedding_tree.append(t_et)
        node_per_keyword_list.extend(node_per_keyword)
    
        TIME_code_total = time.time() - TIME_code_start
        PATH_output_plots = os.path.join(PATH_output_dir, domain)
        if log_time:
            plot_time_analysis(TIME_context_tree, TIME_embedding_tree, TIME_code_total, PATH_output_plots, node_per_keyword_list)

if __name__ == "__main__":
    PATH_output_dir = "/home/ppathak2/Hypothesis_Generation_Active_Learning/output_trees/BATS"
    PATH_keywords_file = "/home/ppathak2/Hypothesis_Generation_Active_Learning/Theoretical Physics.txt"
    # domain = os.path.basename(PATH_keywords_file).split(".")[0]
    

    # with open(PATH_keywords_file, 'r') as f:
    #     keywords = f.readlines()
    keywords = ['actor', 'actress', 'batman', 'batwoman']
    # keywords = ['actor', 'batman', 'boar', 'boy', 'brother', 'buck', 'bull', 
    #             'businessman', 'chairman', 'dad', 'daddy', 'duke', 'emperor',
    #              'father', 'fisherman', 'fox', 'gentleman', 'god', 'grandfather', 
    #              'grandpa', 'grandson', 'groom', 'headmaster', 'heir', 'hero',
    #              'hound', 'husband', 'king', 'lion', 'man', 'manager',
    #              'mister', 'murderer', 'nephew', 'poet', 'policeman', 
    #              'prince', 'ram', 'rooster', 'sculptor', 'sir',
    #              'son', 'stallion', 'stepfather', 'superman', 'tiger',
    #              'uncle', 'valet', 'waiter', 'webmaster']
    domain = "gender noun"
    run(PATH_output_dir, domain, keywords)