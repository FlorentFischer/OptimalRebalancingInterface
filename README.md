# Optimal Rebalancing Interface

This GitHub repository is constitued of the code structure necessary for optimal rebalancing dashboard deployment (py file for the webapp deployment and ipynb file for a local dashboard deployment), a file with all libraries required and this ReadMe file that gives you indications about this repository.



_____________________________________________________________________________________________________
# DashBoard Using:

## 1 - Interactive part

> 1.1 - Portfolio Selection
  
>> Portfolio selection: Here you need to choose which portfolio you want to use among three possibilities (LPP-40 2000, LPP-40 2005, LPP-40 2015). Two other portfolios are available to use only for the robustness check (LPP-25 2015 and LPP-60 2015). You just need to click on the input box and select one of the 5 portfolios proposed. When you have chosen the portfolio, you will have the same consituents in your portfolio as the related benchmark constituents.   
  
> 1.2 - Method of rebalancing 

>> In this subsection you will have the possibility to choose a rebalancing method among 9 possible methods. You just need to click on the one you want to use. 
  
> 1.3 - Rebalancing value 

>> This final subsection allows you to choose the value you want to use for the rebalancing depending on the method you have chosen. You just need to click on the input box and specified a value for the rebalancing. Here is a small guidance for the values possibilities studied in the project:
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

At the right top of the results presentation, you can have access on some metrics to analyze the performance of your portfolio with the methods and value of rebalancing selected.
The metrics presented are: annual tracking error, annual rebalancing cost value, portfolio annual returns, benchmark annual returns, and annual information ratio of the portfolio. All this metrics depends on the interactive part selection. 


## 3- Plots and charts 

> **Cumulative returns of portfolio and benchmark:**

> This plot show the cumulative returns of the benchmark and also the cumulative returns of the portfolio. This cumulative returns plot include the trasaction costs when a portfolio rebalancing occurs. 

> **Portfolio weights variations:**

> This plot will show you the portfolio weights of each assets in the portfolio. It allows you to highlight when portfolio rebalancing occurs with small steep pikes in allocations that reach the initial optimal allocation.

> **Benchmark weights allocation:**

> This pie chart plot will show you the benchmark weight allocation, the weights defined as optimal according to pictet LPP products. For general LPP-40 the repartition provided is 40% stocks and 60% bonds. You can click on a precise sector if you want to zoom on their components. 

> **Monthly returns of the portfolio and the benchmark:**

> On this barplot you can see the monthly returns of the benchmark and also the monthly returns of the portfolio. So you can easily compare them


# Jupyter NoteBook for local deployment:




# Webapp Dashboard Hosting:






