# Optimal Rebalancing Interface

This GitHub repository is constitued of the code structure necessary for optimal rebalancing dashboard deployment (.py file for the webapp deployment and .ipynb file for a local dashboard deployment), a file with all libraries required, the project main Notebook, and this ReadMe file that gives you indications about this repository.



_____________________________________________________________________________________________________
# DashBoard Using:

DashBoard link: https://optimalrebalance.pythonanywhere.com/

## 1 - Interactive part

An important element to know is that after any interaction on the dashboard, the updating process works automatically. So no more additional interactions are required to show the results. Depending on the browser you use, the browser tab related to the dashboard must show "Updating...", during the updating process. 

> 1.1 - Portfolio Selection
  
>> Portfolio selection: Here you need to choose which portfolio you want to use among three possibilities (LPP-40 2000, LPP-40 2005, LPP-40 2015). Two other portfolios are available to use only for the robustness check (LPP-25 2015 and LPP-60 2015). You just need to click on the input box and select one of the 5 portfolios proposed. When you have chosen the portfolio, you will have the same constituents in your portfolio as the related benchmark constituents.   
  
> 1.2 - Method of rebalancing 

>> In this subsection you will have the possibility to choose a rebalancing method among 9 possible methods. You just need to click on the one you want to use. 
  
> 1.3 - Rebalancing value 

>> This final subsection allows you to choose the value you want to use for the rebalancing depending on the method you have chosen. You just need to click on the input box and specify a value for the rebalancing. Here is a small guide for the possibilities of the value studied in the project:
>>> No-rebalancing: Take no values for rebalancing 


>>> Fixed interval rebalancing: Take integer values for the rebalancing, in a range of 1 to 36 or over. (limit is the period length) 


>>> Absolute weights deviation: Take decimal values for the rebalancing, in a range of [0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1]


>>> Relative weights deviation: Take decimal values for the rebalancing, in a range of [0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1]


>>> Tracking Error deviation: Take decimal values for the rebalancing, in a range of [0; 0.0025; 0.005; 0.0075; 0.01; 0.0125; 0.015; 0.0175; 0.02; 0.0225; 0.025; 0.0275; 0.03; 0.0325; 0.035]


>>> Tracking Error and cost deviation: Take decimal values for the rebalancing in a range of [0; 0.0025; 0.005; 0.0075; 0.01; 0.0125; 0.015; 0.0175; 0.02; 0.0225; 0.025; 0.0275; 0.03; 0.0325; 0.035]


>>> Momentum 4-months: Take decimal values for the rebalancing in a range of [0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7; 1.8; 1.9]


>>> Momentum 7-months: Take decimal values for the rebalancing in a range of [0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7; 1.8; 1.9]


>>> Momentum 13-months: Take decimal values for the rebalancing in a range of [0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 1.1; 1.2; 1.3; 1.4; 1.5; 1.6; 1.7; 1.8; 1.9]



## 2- Metrics Overview

At the right top of the presentation of the results, you can have access to some metrics to analyze the performance of your portfolio with the methods and value of rebalancing selected.
The metrics presented are annual tracking error, annual rebalancing cost value, portfolio annual returns, benchmark annual returns, and annual information ratio of the portfolio. All these metrics depend on the interactive part selection. 


## 3- Plots and charts 

> **Cumulative returns of portfolio and benchmark:**

> This plot shows the cumulative returns of the benchmark and also the cumulative returns of the portfolio. This cumulative returns plot includes the transaction costs when a portfolio rebalancing occurs. 

> **Portfolio weights variations:**

> This plot will show you the portfolio weights of each asset in the portfolio. It allows you to highlight when portfolio rebalancing occurs with small steep pikes in allocations that reach the initial optimal allocation.

> **Benchmark weights allocation:**

> This pie chart plot will show you the benchmark weight allocation, the weights defined as optimal according to Pictet LPP products. For general LPP-40 the repartition provided is 40% stocks and 60% bonds. You can click on a precise sector if you want to zoom in on their components. 

> **Monthly returns of the portfolio and the benchmark:**

> On this barplot you can see the monthly returns of the benchmark and also the monthly returns of the portfolio. So you can easily compare them



# Jupyter NoteBook for local deployment:

The easiest way to deploy the dashboard locally is to use an IDE such as Jupyter Notebook. You will just have to launch the "OPR_Dashboard.ipynb" code file in your Jupyter environment without forgetting to include the right dataset path in the data import part. Then you just have to launch the cells one after one. When the last cell is launched, a local deployment link should appear. You will just have to click on it to have local access to the dashboard functionality.  

Required libraries for local deployment are available in the file: "requirements.txt"


# Webapp Dashboard Hosting:

To deploy the dashboard globally on the internet, we used the site "https://www.pythonanywhere.com/" which allows to host a web site built in python. Concerning the technical part of this global deployment, we have only used the flask library on pythonanywhere, and we also implemented some technical manipulation on the pythonanywhere terminal but we will not go on precise details for this kind of manipulations. For this global deployment, we have used the same dataset but with a modified format ".xls" and not the ".xlsx" one. We have also used the file "OPR_Dashboard.py"



# Project work: 

We have also put at your disposal in this GitHub repository the Jupyter Notebook of the research work. It is the main Jupyter Notebook that contains every precise part of the project and provides all results of the project. The file name is the following one: "QARM_II_Project_Work(Group3).ipynb"

Required libraries for project Notebook are available in the file: "requirements.txt"

You can also have access to this project Notebook on Google Colab: https://colab.research.google.com/drive/1TS-nNZNYLjLJ5VLVFxpeYcwLsh4G22Nl?usp=sharing


















