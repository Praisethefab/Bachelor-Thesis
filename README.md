# Bachelor-Thesis
A repository containing data and code for my bachelor thesis. <br>

## Structure
The "experimental setting" directory contains the code to reproduce the experiments from the thesis. <br>
The "rankings" directory contains the .ods files containing manually created tables to calculate the rankings of the calibrators based on AUV. <br>
The "value_analysis_binary.ipynb" can be used to analyse the value of your own models. <br>

## How to use "experimental setting"
Download the directory. <br>
run "pip install -r requirements.txt" inside of it. <br>
run "main_tabulardata.py" by modifying the const parameters as you see fit, it will log the results in a file you named and create graphs for all random states you selected. <br>
<br>
"value_graph.py", creates the tables and graphs related to value, to run this you must have run at some point "main_tabulardata.py" with CALCULATE_VALUE = True. <br>
"griddify_Value.py", converts the graphs related to value into grids, to run this you must have first run "value_graph.py". <br>
"convert_to_table.py", creates the tables and graphs not related to value. <br>
"table columnizer.py", converts all the tables into columns, to run this you must have first run "convert_to_table.py", if you have not run "value_graph.py" before at some point the program will crash but all other tables will be columnized. <br>
"histograms.py", creates the confusion matrices as histograms and turns them into grids. <br>
"griddify_Curve.py", converts the ROC curves and calibration curves into grids based on random state. <br>
"griddify_Diagram by model.py", converts the reliability diagrams into grids based on model. <br>
"griddify_Diagram.py", converts the reliability diagrams into grids based on random state. <br>
"griddify_roc_by_model.py", converts the ROC curves into grids based on model. <br>
"ReliabilityDiagram.py", this is used inside "main_tabulardata.py" and must not be run. <br>
