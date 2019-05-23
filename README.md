# Thesis
This is my master's thesis

Doc		The folder containing all of the documentation for all of the functions in the library
Library		Contains all of the scripts that are used repeatedly
Experiment	Contains all of the content of the actual experiment

To run the experiment, run these scripts in this order
# create the repo list to crawl
Experiment/Data/repo_list.py
# crawl github for all of the comments
Experiment/Data/comments_for_pull_reqs.py
# aggregate the text from the produced jsons into a repo-level json for each repository
Experiment/Data/AggregateText.py
# clean the text from markup, etc.
Experiment/Data/CleanText.py
# process the text for the big 5 and symlog values
Experiment/Data/ProcessedText.py
# aggregate the symlog metrics for analysis
Experiment/Analysis/SYMLOG_metrics.py
# aggregate the big5 metrics for analysis
Experiment/Analysis/Big5_metrics.py
# aggregate all of the disparate metrics into a single directory (experiment_data) for analysis 
Experiment/Analysis/experiment_data.py
# finally, run the analysis
Experiment/Analysis/analysis.py
