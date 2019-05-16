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
Data/CleanText.py
