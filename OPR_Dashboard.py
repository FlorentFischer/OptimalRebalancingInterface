#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output


# In[62]:


import matplotlib.pyplot as plt

from matplotlib import cycler
colors = cycler('color',
                ['#669FEE', '#66EE91', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('figure', facecolor='#313233')
plt.rc('axes', facecolor="#313233", edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors,
       labelcolor='gray')
plt.rc('grid', color='474A4A', linestyle='solid')
plt.rc('xtick', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('legend', facecolor="#313233", edgecolor="#313233")
plt.rc("text", color="#C9C9C9")
plt.rc('figure', facecolor='#313233')


# In[63]:


# Database loading 
lpp2000 = pd.read_excel("LPP2000.xlsx",sheet_name='LPP2000', index_col="Date", parse_dates=True)
lpp2005 = pd.read_excel("LPP2000.xlsx",sheet_name='LPP2005', index_col="Date", parse_dates=True)
lpp2015 = pd.read_excel("LPP2000.xlsx",sheet_name='LPP2015', index_col="Date", parse_dates=True)

# Change dataframe columns names 
lpp2000.columns = ["Swiss Bond", "Swiss Stocks", "Mondial Stocks", "Euro Bonds", "Mondial Bonds"]
lpp2005.columns = ["Swiss Bond", "Mondial Bonds", "Swiss Stocks", "Mondial Stocks","Swiss Real Estate (Other)","World Real Estate (Other)","Hedge funde (Other)","Private Equity (Other)"]
lpp2015.columns = ["Swiss Bond","Dev. Countries Bond","Emerg. Countries Bond","Corporate Bond", "Swiss Stocks", "Mondial Stocks", "Small Cap Stocks", "Real Estate (Other)","Hedge fund (Other)"]

# Define columns of differents assets 
columns_2000 = ["Swiss Bond", "Euro Bonds", "Mondial Bonds", "Swiss Stocks", "Mondial Stocks"]
columns_2005 = ["Swiss Bond", "Mondial Bonds", "Swiss Stocks", "Mondial Stocks","Swiss Real Estate (Other)","World Real Estate (Other)","Hedge funde (Other)","Private Equity (Other)"]
columns_2015 = ["Swiss Bond","Dev. Countries Bond","Emerg. Countries Bond","Corporate Bond", "Swiss Stocks", "Mondial Stocks", "Small Cap Stocks", "Real Estate (Other)","Hedge fund (Other)"]

# Create returns of all LPP
lpp2000 = pd.concat((lpp2000["Swiss Bond"], lpp2000["Euro Bonds"], lpp2000["Mondial Bonds"], lpp2000["Swiss Stocks"], lpp2000["Mondial Stocks"]), axis=1).pct_change(1)
lpp2005 = lpp2005.pct_change(1)
lpp2015 = lpp2015.pct_change(1)

# LPP40 weights given differents years 
weight_LPP_2000_40 = [0.45, 0.1, 0.05, 0.15, 0.25]
weight_LPP_2005_40 = [0.3,0.2,0.1,0.2,0.05,0.05,0.05,0.05]
weight_LPP_2015_40 = [0.3,0.1,0.05,0.05,0.15,0.2,0.05,0.05,0.05]


lpp40_2000 = np.multiply(lpp2000, weight_LPP_2000_40).sum(axis=1)
lpp40_2005 = np.multiply(lpp2005, weight_LPP_2005_40).sum(axis=1)
lpp40_2015 = np.multiply(lpp2015, weight_LPP_2015_40).sum(axis=1)


lpp2000["LPP 40"] = lpp40_2000
lpp2005["LPP 40"] = lpp40_2005
lpp2015["LPP 40"] = lpp40_2015

lpp2000 = lpp2000.dropna()
lpp2005 = lpp2005.dropna()
lpp2015 = lpp2015.dropna()

# LPP-25 2015 dataframe (Robustness check)
weight_LPP_2015_25 = [0.45,0.1,0.05,0.05,0.1,0.15,0.0,0.05,0.05]
lpp25_2015 = np.multiply(lpp2015[columns_2015], weight_LPP_2015_25).sum(axis=1)
lpp2015["LPP 25"] = lpp25_2015
lpp2015 = lpp2015.dropna()


# LPP-60 2015 dataframe (Robustness check)
weight_LPP_2015_60 = [0.1,0.1,0.05,0.05,0.20,0.30,0.10,0.05,0.05]
lpp60_2015 = np.multiply(lpp2015[columns_2015], weight_LPP_2015_60).sum(axis=1)
lpp2015["LPP 60"] = lpp60_2015
lpp2015 = lpp2015.dropna()


# In[70]:


class Rebalancing:
  """
  Database: Dataframe
  Benchmark: String (column name)
  Columns: List of string containning the name of the asset
  Weight_ben: list of weight
  Cost : initially set at 0.01, it can be changed when using the class function.
         This cost correspond to a round trip rebalancing case (including buying
         at the begining and selling at the end cost)


  List of methods:
  - No Rebalancing
  - Fixed Interval Rebalancing
  - Absolute Deviation
  - Relative Deviation
  - Tracking Error Deviation
  - Tracking Error Upgrade
  - Momentum strategy 4mth (TAA)
  - Momentum strategy 7mth (TAA)
  - Momentum strategy 13mth (TAA)

  """

  def __init__(self, database, benchmark, columns, weight_ben, cost=0.01):
    # INPUTS
    self.database = database.dropna()
    self.benchmark = benchmark
    self.weight_ben = weight_ben
    self.columns = columns
    self.cost = cost

    if np.array(self.weight_ben).sum()!=1:
      print("WARNINGS: SUM OF THE CAPITAL DIFFERENT OF 1")

    # VARIABLES
    self.weight = None
    self.returns = None
    self.portfolio = None
    self.cost_date = list()
    



  def no_rebalancing(self):
    # COMPUTE THE WEIGHT
    absolute_weight =  np.multiply((1 + self.database[self.columns].cumsum()),self.weight_ben)
    self.weight = np.divide(absolute_weight, absolute_weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.cost_date.append(self.database.index[0])
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    #Parameter
    self.param = 'No parameter'



  def fixed_interval_rebalancing(self, n):
    """
    n = number of month between each rebalancing 
    """
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()

    # Block any to low values that induce issues   

    for i in range(0,len(self.database)):

      if i%n == 0:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose() 

        weights.append(current_weight)

        self.cost_date.append(self.database.index[i])

      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)


    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = n

  def absolute_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    self.cost_date.append(self.database.index[0])

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      deviation = np.abs(current_weight_array-ben_weight_array).sum()

      if deviation > pct:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)

        self.cost_date.append(self.database.index[i])
      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct

  def relative_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    self.cost_date.append(self.database.index[0])

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      deviation = np.abs((current_weight_array-ben_weight_array).sum())

      if deviation>pct:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)


    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def tracking_error_deviation(self,pct):

    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    self.cost_date.append(self.database.index[0])

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      
      returns_pf = np.multiply(self.database[self.columns].iloc[start:i,:],
                               current_weight_array).sum(axis=1)
      
      returns_ben = np.multiply(self.database[self.columns].iloc[start:i,:],
                               self.weight_ben).sum(axis=1)
      if i-start>1:
        tracking_error = np.std(returns_ben-returns_pf)

      else:
        tracking_error = 0

      if tracking_error>pct:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
      
      else:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def tracking_error_cost_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    n = 1
    start = 0

    self.cost_date.append(self.database.index[0])

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      
      returns_pf = np.multiply(self.database[self.columns].iloc[start:i,:],
                               current_weight_array).sum(axis=1)
      
      returns_ben = np.multiply(self.database[self.columns].iloc[start:i,:],
                               self.weight_ben).sum(axis=1)
      if i-start>1:
        tracking_error = np.std(returns_ben-returns_pf)

      else:
        tracking_error = 0

      deviation = np.abs(current_weight_array-ben_weight_array).sum()
      

      if tracking_error>pct and returns_pf.sum() - self.cost * len(self.columns) * n > 0:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
        n = n + 1
      
      else:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)


    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def momentum4(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    self.cost_date.append(self.database.index[0])


    for i in range(0,len(self.database)):

      current_weight_array = np.array(current_weight)

      returns = np.multiply(self.database[self.columns].iloc[i-1:i,:],
                               current_weight_array).sum(axis=1)

      current_returns = returns.sum()

      try:
        returns_pf = np.multiply(self.database[self.columns].iloc[i-3:i,:],
                                np.array(weights[-1].values[-4:,:])).sum(axis=1)
                                
        rol = returns_pf.mean()

      except:

        rol=0


      if rol*(1 + pct) <= current_returns and current_returns > 0 and rol > 0:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)


    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def momentum7(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    self.cost_date.append(self.database.index[0])


    for i in range(0,len(self.database)):

      current_weight_array = np.array(current_weight)

      returns = np.multiply(self.database[self.columns].iloc[i-1:i,:],
                               current_weight_array).sum(axis=1)

      current_returns = returns.sum()

      try:
        returns_pf = np.multiply(self.database[self.columns].iloc[i-6:i,:],
                                np.array(weights[-1].values[-7:,:])).sum(axis=1)
                                
        rol = returns_pf.mean()

      except:
        rol=0


      if rol*(1 + pct) <= current_returns and current_returns > 0 and rol > 0:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)



    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def momentum13(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """
    
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    # Entry cost
    self.cost_date.append(self.database.index[0])

    # To low parameter are not accepted  by the function 

    for i in range(0,len(self.database)):

      current_weight_array = np.array(current_weight)

      returns = np.multiply(self.database[self.columns].iloc[i-1:i,:],
                               current_weight_array).sum(axis=1)

      current_returns = returns.sum()

      try:

        returns_pf = np.multiply(self.database[self.columns].iloc[i-12:i,:],
                                np.array(weights[-1].values[-13:,:])).sum(axis=1)
                                
        rol = returns_pf.mean()

      except:

        rol=0

      if rol*(1 + pct) <= current_returns and current_returns > 0 and rol > 0:

        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()            
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])
      
      else:

        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)
        
        weights.append(current_weight)



    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)

    # Parameter 
    self.param = pct


  def visualisation(self, graphs=True):


    # COMPUTE ANNUAL RETURN
    self.annual_pf = self.portfolio.mean()*12*100
    self.annual_idx = self.database[self.benchmark].mean()*12*100

    # COMPUTE SOME METRICS
    te = (np.std(self.portfolio - self.database[self.benchmark]))*100
    
    self.te = te
    
    self.total_cost = (self.cost * len(self.cost_date) * len(self.columns))*100
    
    self.IR = ((self.portfolio - self.database[self.benchmark]).mean()/(np.std(self.portfolio - self.database[self.benchmark])))*np.sqrt(12)
    


# In[71]:


class Visualization:
  """ Class """

  def __init__(self, database, portfolio, bench, weight, SAA_weight=None):
    """ Ini """
    self.database = database
    self.portfolio = portfolio
    self.bench = bench
    self.weight = weight
    self.saa = SAA_weight
    if self.saa!=None:
        self.pf_saa = np.multiply(database, SAA_weight).sum(axis=1)


  def fig_cumulative_returns(self):
    fig = go.Figure()

    # Add first plot that represent the Cumulative return of the strategie
    fig.add_trace(go.Scatter(x=self.portfolio.index, y=self.portfolio.cumsum().values*100,
                      mode='lines',
                      name="Portfolio", line_color = '#FF801A'))

    # Add second plot that represent the Cumulative return of the Benchmark
    fig.add_trace(go.Scatter(x=self.bench.index, y=self.bench.cumsum().values*100,
              mode='lines',
              name="Benchmark", line_color ='#017F85'))

    if self.saa!=None:
        # Add second plot that represent the Cumulative return of the SAA Benchmark
        fig.add_trace(go.Scatter(x=self.pf_saa.index, y=self.pf_saa.cumsum().values*100,
                  mode='lines',
                  name="SAA"))

    # Add some layout
    fig.update_layout(title="Cumulative Returns (Portfolio and Benchmark)",
                xaxis_title="Years",
                yaxis_title="Cumulative Returns %", title_x=0.5,
                paper_bgcolor="#131313",
                plot_bgcolor="#131313",
                      legend=dict(
                      x=0.03,
                      y=0.97,
                      traceorder='normal',
                      font=dict(size=12)),
                template="plotly_dark")
    return fig



  def fig_weight_portfolio(self):
    fig = go.Figure()

    # Add as many graphic as it is necessary for each weight
    for i, name in zip(np.arange(self.weight.shape[1]), list(self.weight.columns)):
      fig.add_trace(go.Scatter(x=self.weight.index, y=(self.weight.iloc[:,i]*100).values,
                        mode='lines',
                        name=name))


    fig.update_layout(title="Portfolio Weight Variations",
                xaxis_title="Years",
                yaxis_title="Weight %", title_x=0.5,
                paper_bgcolor="#131313",
                plot_bgcolor="#131313",
                      legend=dict(
                      x=0.03,
                      y=0.97,
                      traceorder='normal',
                      font=dict(size=12)),
                template="plotly_dark")
    return fig


  def yearly_return_comparaison(self):
    p = self.portfolio

    def yearly_return_values(p):

        total = 0
        positif = 0


        r=[]
        # Loop on each different year
        for year in p.index.strftime("%y").unique():
            nbm = p.loc[p.index.strftime("%y")==year].index.strftime("%m").unique()
            # Loop on each different month
            for mois in nbm:

                monthly_values =  p.loc[p.index.strftime("%y:%m")=="{}:{}".format(year,mois)]
                sum_ = monthly_values.sum()

                # Verifying that there is at least 75% of the values
                if len(monthly_values)>15:

                    # Computing sum return
                    s = monthly_values.sum()

                    if s>0:
                        positif+=1

                    else:
                        pass

                    total+=1

                else:
                    pass
                r.append(sum_)



        return r
    def yearly_return_index(p):
        total = 0
        positif = 0


        r=[]
        # Loop on each different year
        for year in p.index.strftime("%y").unique():
            e = []
            nbm = p.loc[p.index.strftime("%y")==year].index.strftime("%m").unique()
            # Loop on each different month
            for mois in nbm:

                monthly_values =  p.loc[p.index.strftime("%y:%m")=="{}:{}".format(year,mois)]
                sum_ = monthly_values.sum()

                # Verifying that there is at least 75% of the values
                if len(monthly_values)>15:

                    # Computing sum return
                    s = monthly_values.sum()

                    if s>0:

                        positif+=1

                    else:
                        pass

                    total+=1

                else:
                    pass
                e.append(sum_)
            r.append(e)




        r[0]=[0 for _ in range(12-len(r[0]))] + r[0]

        r =  pd.DataFrame(r,columns=["January","February","March","April","May","June",
                                        "July","August","September","October","November","December"], index=p.index.strftime("%y").unique())

        v = []
        for i in [i for i in r.index]:
            for c in [i for i in r.columns]:
                if r.loc[i,c]!=0:
                    v.append("{}:20{}".format(c,i))
        return v

    portfolio = p*100
    pf_values = yearly_return_values(portfolio)
    pf_index = yearly_return_index(portfolio)

    bench = self.bench*100
    bench_values = yearly_return_values(bench)
    bench_index = yearly_return_index(bench)


    fig = go.Figure()
    fig.add_trace(go.Bar(name="Portfolio", x=pf_index, y=pf_values, marker_line_color="#FF801A",
                          marker_color="#FF801A", opacity=0.9))
    fig.add_trace(go.Bar(name="Benchmark", x=bench_index, y=bench_values, marker_line_color="#017F85",
                          marker_color="#017F85", opacity=0.9))
    # Change the bar mode
    fig.update_layout(barmode='group', title="Portfolio and Benchmark Monthly Returns",yaxis_title="Returns %",
                      paper_bgcolor="#131313",
                    plot_bgcolor="#131313",
                    legend=dict(
                    x=0.03,
                    y=0.97,
                    traceorder='normal',
                    font=dict(
                        size=12,)),
                      title_x=0.5,
                      template="plotly_dark")

    return fig

reb = Rebalancing(lpp2000, "LPP 40", columns_2000, weight_LPP_2000_40, cost=0.0013)
reb.momentum4(0.03)
reb.visualisation()


# In[72]:


benchs = [{'label': "LPP-40 (2000)", 'value': "lpp200040"},
            {'label': "LPP-40 (2005)", 'value': "lpp200540"},
            {'label': "LPP-40 (2015)", 'value': "lpp201540"},
            {'label': "LPP-25 (2015, Robustness check only)", 'value': "lpp201525"},
            {'label': "LPP-60 (2015, Robustness check only)", 'value': "lpp201560"}]


# In[73]:


resume_2000_40 = pd.DataFrame([[45, "Bonds", "Swiss"],
             [10, "Bonds", "Euro"],
             [5, "Bonds", "World"],
             [25, "Stocks", "World"],
             [15, "Stocks", "Swiss"],], columns=["Weight %", "Sector", "Indices"])

resume_2005_40 = pd.DataFrame([[30, "Bonds", "Swiss"],
             [20, "Bonds", "World"],
             [10, "Stocks", "Swiss"],
             [20, "Stocks", "World"],
             [5, "Stocks", "Swiss Real Estate"],
             [5, "Stocks", "World Real Estate"],
             [5, "Other", "Hedge fund"],
             [5, "Other", "Private equity"]], columns=["Weight %", "Sector", "Indices"])

resume_2015_40 = pd.DataFrame([[30, "Bonds", "Swiss"],
             [5, "Bonds", "Corporate"],
             [10, "Bonds", "Dev. Countries "],
             [5, "Bonds", "Emerg. Countries"],
             [15, "Stocks", "Swiss"],
             [20, "Stocks", "World"],
             [5, "Stocks", "Small Cap."],
             [5, "Other", "Swiss Real Estate"],
             [5, "Other", "Hedge fund"]], columns=["Weight %", "Sector", "Indices"])

resume_2015_25 = pd.DataFrame([[45, "Bonds", "Swiss"],
             [10, "Bonds", "Corporate"],
             [5, "Bonds", "Dev. Countries "],
             [5, "Bonds", "Emerg. Countries"],
             [10, "Stocks", "Swiss"],
             [15, "Stocks", "World"],
             [0, "Stocks", "Small Cap."],
             [5, "Other", "Swiss Real Estate"],
             [5, "Other", "Hedge fund"]], columns=["Weight %", "Sector", "Indices"])

resume_2015_60 = pd.DataFrame([[10, "Bonds", "Swiss"],
             [10, "Bonds", "Corporate"],
             [5, "Bonds", "Dev. Countries "],
             [5, "Bonds", "Emerg. Countries"],
             [20, "Stocks", "Swiss"],
             [30, "Stocks", "World"],
             [10, "Stocks", "Small Cap."],
             [5, "Other", "Swiss Real Estate"],
             [5, "Other", "Hedge fund"]], columns=["Weight %", "Sector", "Indices"])


# In[74]:


params = {"font-size":"20px",
             "margin-left":"5px",
              "margin-top":"15px",
             "background-color":"#131313",
             "border-radius": "5px",
              "height": "75px"
             }
te = html.Div([html.Div([dcc.Markdown("", id="te")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Tracking Error")], style={"margin-left":"5px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })

cost = html.Div([html.Div([dcc.Markdown("", id="cost")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Cost")], style={"margin-left":"5px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"5px",
                          "border-radius": params["border-radius"],
                            })


ret_pf = html.Div([html.Div([dcc.Markdown("", id="ret_pf")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Portfolio returns")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"],
                            })

ret_idx = html.Div([html.Div([dcc.Markdown("", id="ret_idx")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Benchmark returns")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"],
                            })

IR = html.Div([html.Div([dcc.Markdown("", id="IR")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Information Ratio")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"]})


# In[75]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {"background-color":"#131313",
         "color":"#ffffff"}



########## HEADER
header = html.Div([dcc.Markdown("**OPTIMAL REBALANCING METHODS **", style={"font-size":"35px"}),
                   dcc.Markdown("", style={"font-size":"15px"})],

                  style={"padding": "50px",
                        "background":"#131313",
                        "margin":"-15px 0px -0px -10px",
                        "textAlign":"center",
                        "color":"#FFFFFF"})

########## BANDEAU 1
lpp = html.Div([dcc.Markdown("**PORTFOLIO SELECTION**", style={"color":colors["color"]}),
                dcc.Dropdown(id="actifs",
                     options=benchs,
                    value="lpp200040",
                     multi=False)], style={"margin":"0px 0px 0px 0px", "width":"60%",
                                           "padding":"30px 0px 0px 0px",'marginLeft' : '50px'})

input_value = html.Div([dcc.Markdown("**VALUE FOR REBALANCING [(see documentation)] (https://github.com/FlorentFischer/OptimalRebalancingInterface.git)**", style={"color":colors["color"],"margin":"0px 0px 0px 50px"}),
                        dcc.Input(id="value", type="text", placeholder="ex: 0.01 (for %) or 12 (for months)", style={"margin":"0px 0px 0px 50px","width":"35%"})])

inputs = html.Div([dcc.Markdown("**VALUE FOR REBALANCING (see the documentation) **", style={"color":colors["color"]}),
                  lpp, input_value], style={"columnCount":2,"margin":"0px 0px 0px 50px"})

methods = html.Div([dcc.Markdown("**METHOD OF REBALANCING**", style={}),
                    dcc.RadioItems(id="Optimizor",options=[{'label': "No rebalancing", 'value': "NR"},
                                            {'label': "Fixed interval rebalancing", 'value': "FIR"},
                                            {'label': "Absolute weights deviation ", 'value': "AD"},
                                            {'label': "Relative deviation", 'value': "RD"},
                                            {'label': "Tracking Error deviation", 'value': "TE"},
                                            {'label': "Tracking Error and cost deviation", 'value': "TEU"},
                                            {'label': "Momentum strategy (4mth)", 'value': "MOM3"},
                                            {'label': "Momentum strategy (7mth)", 'value': "MOM6"},
                                            {'label': "Momentum strategy (13mth)", 'value': "MOM12"}],value="NR")],
                   style={"color":"#ffffff","margin":"25px 0px 25px 50px"})

spaces = html.Div([dcc.Markdown("", style={"height":"100px"})])


review = html.Div([te, cost, ret_pf, ret_idx,IR], style={"columnCount":5, "background-color":"#131313",
                                                         "margin":"0px 0px -0px 0px", "color":"white",
                                                         "text-align": "center"})


bandeau_1_gauche = html.Div([lpp, methods, input_value, spaces], style={"margin":"0px 0px 0px 0px",
                                                                        "background-color":colors["background-color"],
                                                                       "padding":"0px 0px 0px 0px"})
bandeau_1_droite = html.Div([review,dcc.Graph(id="cumret", style={"margin":"0px 0px 0px 0px"})])

bandeau_1 = html.Div([bandeau_1_gauche, bandeau_1_droite], style={"columnCount":2,"height":"700px","margin":"20px 0px -0px -0px"})

########## BANDEAU 2

bandeau_2 = html.Div([dcc.Graph(id="weight"),
                     dcc.Graph(id="drawdown")], style={"columnCount":2,
                                                      "margin":"-135px 0px 0px 0px"})

########## BANDEAU 3
bandeau_3 = html.Div([dcc.Graph(id="returns")], style={"margin":"30px 0px 0px 0px"})

########## Dashboard
dashboard = html.Div([header,
                     bandeau_1,
                     bandeau_2,
                     bandeau_3], style={"background":"#303030",
                                 "margin":"0px -15px -0px -15px"
                                       })

app.layout = dashboard

####### CALLBACKS

@app.callback(Output("cumret", "figure"),
              Output("weight", "figure"),
              Output("drawdown", "figure"),
              Output("returns", "figure"),
              Output("cost", "children"),
              Output("te", "children"),
              Output("ret_pf", "children"),
              Output("ret_idx", "children"),
              Output("IR", "children"),
              Input("value", "value"),
              Input("Optimizor", "value"),
             Input("actifs","value"))

def affichage(value, method, lpp_value):
    print(value, method)


    if lpp_value == "lpp200040":
        database, weight, bench, columns = lpp2000, weight_LPP_2000_40, "LPP 40", columns_2000
        exp, title, saa_weight ="40%","LPP-40 (2000) Target Weight Allocation", None
        cost_ = 0.0035
        resume = resume_2000_40

    elif lpp_value == "lpp200540":
        database, weight, bench, columns = lpp2005, weight_LPP_2005_40, "LPP 40", columns_2005
        exp, title, saa_weight ="30%", "LPP-40 (2005) Target Weight Allocation", None
        cost_ = 0.0075
        resume = resume_2005_40

    elif lpp_value == "lpp201540":
        database, weight, bench, columns = lpp2015, weight_LPP_2015_40, "LPP 40", columns_2015
        exp, title, saa_weight ="30%", "LPP-40 (2015) Target Weights Allocation", None
        cost_ = 0.0065
        resume = resume_2015_40
        
    elif lpp_value == "lpp201525":
        database, weight, bench, columns = lpp2015, weight_LPP_2015_25, "LPP 25", columns_2015
        exp, title, saa_weight ="30%", "LPP-25 (2015) Target Weights Allocation", None
        cost_ = 0.0065
        resume = resume_2015_25
        
    elif lpp_value == "lpp201560":
        database, weight, bench, columns = lpp2015, weight_LPP_2015_60, "LPP 60", columns_2015
        exp, title, saa_weight ="30%", "LPP-60 (2015) Target Weights Allocation", None
        cost_ = 0.0065
        resume = resume_2015_60       
        

    if value == "":
        value = 0

    if value==None:
        value = 0

    reb = Rebalancing(database, bench, columns, weight, cost=cost_)
    if method=="NR":
        reb.no_rebalancing()

    elif method=="FIR":
        if float(value)<1:
            value = 1
            reb.fixed_interval_rebalancing(int(value))
        else:
            reb.fixed_interval_rebalancing(int(value))

    elif method=="AD":
        reb.absolute_deviation(float(value))

    elif method=="RD":
        reb.relative_deviation(float(value))

    elif method=="TE":
        reb.tracking_error_deviation(float(value))

    elif method=="TEU":
        reb.tracking_error_cost_deviation(float(value))

    elif method=="MOM3":
        reb.momentum4(float(value))

    elif method=="MOM6":
        reb.momentum7(float(value))

    else:
        reb.momentum13(float(value))


    reb.visualisation()

    vis = Visualization(reb.database[reb.columns], reb.portfolio, reb.database[reb.benchmark], reb.weight, SAA_weight=saa_weight)

    # Metrics computation 
    cum = vis.fig_cumulative_returns()
    weight = vis.fig_weight_portfolio()
    ret = vis.yearly_return_comparaison()
    annual_cost = np.round(reb.total_cost/(len(lpp2000)/12),2)
    cost = "{}%".format(annual_cost)
    annual_te = np.round(reb.te*np.sqrt(12),2)
    te = "{}%".format(annual_te)
    IR_annual =np.round(reb.IR,2)
    IR = "{}".format(IR_annual)


    sunburst_2015_40 = px.sunburst(resume, path=['Sector', 'Indices'], values="Weight %",height = 450,color = 'Sector',
                                  color_discrete_map={'Stocks':'#017F85', 'Bonds':'#FF801A', 'Other':'#07474f'})
    #___________________________________________________________________________________________________
    sunburst_2015_40.update_traces(textfont=dict(color = 'white'),insidetextorientation = 'horizontal')


    #___________________________________________________________________________________________________
    sunburst_2015_40.update_layout(template="plotly_dark", font={"color":"white"},title=title, title_x=0.5,
                paper_bgcolor="#131313",
                plot_bgcolor="#131313",
                      legend=dict(
                      x=0.03,
                      y=0.97,
                      traceorder='normal',
                      font=dict(size=12)))

    sunburst_2015_40.add_trace(go.Sunburst(
        insidetextorientation='radial'))

    # Returns computation 
    tot_ret = np.round(reb.annual_pf,2)
    pf_ret = "{}%".format(tot_ret)
    tot_idx = np.round(reb.annual_idx,2)
    pf_idx = "{}%".format(tot_idx)
    return cum, weight, sunburst_2015_40, ret, cost,te,pf_ret, pf_idx,IR

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:





# In[ ]:




