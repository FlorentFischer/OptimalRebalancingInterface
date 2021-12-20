# OptimalRebalancingInterface

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


>>> Absolute weights deviation: Take decimal values for the rebalancing, in a range of [0 ; 0.1 ; 0.2 ; 0.3 ; 0.4 ; 0.5 ; 0.6 ; 0.7 ; 0.8 ; 0.9 ; 1]


>>> Relative weights deviation: Take decimal values for the rebalancing, in a range of [0 ; 0.1 ; 0.2 ; 0.3 ; 0.4 ; 0.5 ; 0.6 ; 0.7 ; 0.8 ; 0.9 ; 1]


>>> Tracking Error deviation: Take decimal values for the rebalancing


>>> Tracking Error and cost deviation: Take decimal values for the rebalancing


>>> Momentum 4-months: Take decimal values for the rebalancing


>>> Momentum 7-months: Take decimal values for the rebalancing


>>> Momentum 13-months: Take decimal values for the rebalancing



## 2- Metrics Overview

At the right top of the results presentation, you can have access on some metrics to analyze the performance of your portfolio with the methods and value of rebalancing selected.
The metrics presented are: annual tracking error, annual rebalancing cost value, portfolio annual returns, benchmark annual returns, and annual information ratio of the portfolio. All this metrics depends on the interactive part selection. 


## 3- Plots and charts 



# Jupyter NoteBook for local deployment:




# Webapp Dashboard Hosting:






