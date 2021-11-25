#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output


# In[2]:


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


# In[3]:


def describe(lpp):
  """
  Documentation
  """

  # Usual description
  resume = lpp.describe()

  # Non usual description
  sharpe = lpp.mean(axis=0) / lpp.std(axis=0)
  sharpe = pd.DataFrame(sharpe, index=lpp.columns, columns = ["sharpe"]).transpose()

  # Concat the descriptions
  description = pd.concat((resume, sharpe), axis=0)

  return description


# In[4]:
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
##############################################################
##############################################################
##############################################################
#################### Belek au file Path ######################
##############################################################
##############################################################
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

lpp2000 = pd.read_excel("/home/OptimalRebalance/mysite/LPP2000.xls", index_col="Date", parse_dates=True)
lpp2000.columns = ["Swiss Bond", "Swiss Stocks", "Mondial Stocks", "Euro Bonds", "Mondial Bonds"]
columns = ["Swiss Bond", "Euro Bonds", "Mondial Bonds", "Swiss Stocks", "Mondial Stocks"]
lpp2000 = pd.concat((lpp2000["Swiss Bond"], lpp2000["Euro Bonds"], lpp2000["Mondial Bonds"], lpp2000["Swiss Stocks"], lpp2000["Mondial Stocks"]), axis=1).pct_change(1)

weight_LPP2000_25 = [0.6, 0.1, 0.05, 0.1, 0.15]
weight_LPP2000_40 = [0.45, 0.1, 0.05, 0.15, 0.25]
weight_LPP2000_60 = [0.25, 0.1, 0.05, 0.2, 0.4]

lpp25_2000 = np.multiply(lpp2000, weight_LPP2000_25).sum(axis=1)
lpp40_2000 = np.multiply(lpp2000, weight_LPP2000_40).sum(axis=1)
lpp60_2000 = np.multiply(lpp2000, weight_LPP2000_60).sum(axis=1)

lpp2000["LPP 25"] = lpp25_2000
lpp2000["LPP 40"] = lpp40_2000
lpp2000["LPP 60"] = lpp60_2000

lpp2000 = lpp2000.dropna()
describe(lpp2000)


# In[5]:


class Rebalancing:
  """
  Database: Dataframe
  Benchmark: String (column name)
  Columns: List of string containning the name of the asset
  Weight_ben: list of weight

    List of methods:
  - No Rebalancing
  - Fixed Interval Rebalancing
  - Absolute Deviation
  - Relative Deviation
  - Tracking Error Deviation
  - Momentum startegy (TAA)

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



  def fixed_interval_rebalancing(self, n):
    """
    n = number of month between each rebalancing
    """
    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()

    for i in range(0,len(self.database)):
      if i%n!=0:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                 index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)


  def absolute_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)


    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      deviation = np.abs(current_weight_array-ben_weight_array).sum()

      if deviation<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)





  def relative_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)


    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      deviation = np.abs((current_weight_array-ben_weight_array).sum())

      if deviation<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)



  def tracking_error_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

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

      if tracking_error<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)




  def standard_deviation_deviation(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)

      returns_pf = np.multiply(self.database[self.columns].iloc[i-11:i,:],
                               current_weight_array).sum(axis=1)

      std = np.std(returns_pf)
      if std<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)



  def momentum(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)

    start = 0

    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      try:
        returns_pf = np.abs(np.multiply(self.database[self.columns].iloc[i-11:i,:],
                                np.array(weights[-1].values[-12:,:])).sum(axis=1))

        rol = returns_pf.mean()
      except:
        rol=0
      if rol<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)



  def bands_rebalancing(self,pct):
    """
    pct = pourcentage de deviation des poids du portfeuille
    """

    self.cost_date = list()
    current_weight = self.weight_ben
    weights = list()
    ben_weight_array = np.array(self.weight_ben)


    for i in range(0,len(self.database)):
      current_weight_array = np.array(current_weight)
      deviation = np.abs(current_weight_array-ben_weight_array).sum()

      if deviation<pct:
        current_weight = np.multiply(1 + self.database[self.columns].iloc[i:i+1,:],
                                    current_weight)

        weights.append(current_weight)

      else:
        current_weight = pd.DataFrame(self.weight_ben, columns=[self.database.index[i]],
                                index = self.columns).transpose()
        weights.append(current_weight)
        self.cost_date.append(self.database.index[i])

    self.weight = pd.concat(tuple(weights),axis=0)
    self.weight = np.divide(self.weight,self.weight.sum(axis=1).values.reshape(-1,1))

    # COMPUTE THE RETURNS
    self.returns = np.multiply(self.database[self.columns], self.weight)
    self.returns.loc[self.cost_date] = self.returns.loc[self.cost_date] - self.cost

    # COMPUTE THE PORTFOLIO
    self.portfolio = self.returns.sum(axis=1)


  def visualisation(self, graphs=True):


    # COMPUTE ANNUAL RETURN
    self.annual_pf = self.portfolio.mean()*12*100
    self.annual_idx = self.database[self.benchmark].mean()*12*100

    # COMPUTE SOME METRICS
    self.te = (np.std(self.portfolio - self.database[self.benchmark]) * 100)
    self.total_cost = (self.cost * len(self.cost_date) * len(self.columns))*100



# In[15]:




class Visualization:
  """ Class """

  def __init__(self, portfolio, bench, weight):
    """ Ini """
    self.portfolio = portfolio
    self.bench = bench
    self.weight = weight


  def fig_cumulative_returns(self):
    fig = go.Figure()

    # Add first plot that represent the Cumulative return of the strategie
    fig.add_trace(go.Scatter(x=self.portfolio.index, y=self.portfolio.cumsum().values*100,
                      mode='lines',
                      name="Portfolio"))

    # Add second plot that represent the Cumulative return of the Benchmark
    fig.add_trace(go.Scatter(x=self.bench.index, y=self.bench.cumsum().values*100,
              mode='lines',
              name="Bench"))

    # Add some layout
    fig.update_layout(title="Cumulative Returns %",
                xaxis_title="Times",
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


    fig.update_layout(title="Weights %",
                xaxis_title="Times",
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


  def fig_drawdown(self):
    cum_rets = self.portfolio.cumsum()+1

    # Calculate the running maximum
    running_max = np.maximum.accumulate(cum_rets.dropna())

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    drawdown = (cum_rets)/running_max - 1
    self.drawdown = (cum_rets)/running_max - 1



    fig = go.Figure()


    y = np.concatenate((drawdown,np.array([0 for i in range(len(drawdown))])),axis=0)*100


    fig.add_trace(go.Scatter(
        x=drawdown.index.append(drawdown.index[::-1]), y=y,
        fill='toself',
        fillcolor="#C71C1C",
        line_color="#C71C1C",
        name="drawdown",
    ))

    fig.update_traces(mode='lines')

    fig.update_layout(title="Test set drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown %",
                      title_x=0.5,
                      legend=dict(
                    x=0.03,
                    y=0.97,
                    traceorder='normal',
                    font=dict(
                        size=12,)), template="plotly_dark"
                      )
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
    fig.add_trace(go.Bar(name="Portfolio", x=pf_index, y=pf_values, marker_line_color="#505B85",
                          marker_color="#508561", opacity=0.9))
    fig.add_trace(go.Bar(name="Bench", x=bench_index, y=bench_values, marker_line_color="#855061",
                          marker_color="#855061", opacity=0.9))
    # Change the bar mode
    fig.update_layout(barmode='group', title="Portfolio Vs Benchmark returns",yaxis_title="Returns %",
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

reb = Rebalancing(lpp2000, "LPP 40", columns, weight_LPP2000_40, cost=0.0013)
reb.momentum(0.03)
reb.visualisation()

vis = Visualization(reb.portfolio, reb.database[reb.benchmark], reb.weight)
fig = vis.yearly_return_comparaison()
fig.show()


# In[7]:


benchs = [
 {'label': "LPP 2000 25", 'value': "lpp200025"},
 {'label': "LPP 2000 40", 'value': "lpp200040"},
 {'label': "LPP 2000 60", 'value': "lpp200060"},
 {'label': "LPP 2005 25", 'value': "lpp200525"},
 {'label': "LPP 2005 40", 'value': "lpp200540"},
 {'label': "LPP 2005 60", 'value': "lpp200560"},
 {'label': "LPP 2015 25", 'value': "lpp201525"},
 {'label': "LPP 2015 40", 'value': "lpp201540"},
 {'label': "LPP 2015 60", 'value': "lpp201560"}]


# In[8]:


resume_2000_60 = pd.DataFrame([[25, "Bonds", "Swiss"],
             [10, "Bonds", "Euro"],
             [5, "Bonds", "World"],
             [40, "Stocks", "World"],
             [20, "Stocks", "Swiss"],], columns=["Weight %", "Sector", "Indices"])

resume_2000_40 = pd.DataFrame([[45, "Bonds", "Swiss"],
             [10, "Bonds", "Euro"],
             [5, "Bonds", "World"],
             [25, "Stocks", "World"],
             [15, "Stocks", "Swiss"],], columns=["Weight %", "Sector", "Indices"])

resume_2000_25 = pd.DataFrame([[60, "Bonds", "Swiss"],
             [10, "Bonds", "Euro"],
             [5, "Bonds", "World"],
             [15, "Stocks", "World"],
             [10, "Stocks", "Swiss"]], columns=["Weight %", "Sector", "Indices"])


# In[9]:


params = {"font-size":"20px",
             "margin-left":"5px",
              "margin-top":"15px",
             "background-color":"#161616",
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
                       html.Div([dcc.Markdown("Ret Pf.")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"],
                            })

ret_idx = html.Div([html.Div([dcc.Markdown("", id="ret_idx")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Ret Bench.")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"],
                            })

exp = html.Div([html.Div([dcc.Markdown("", id="exp")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Exp monetaire")], style={"margin-left":"0px"})],

                      style={
                            "background-color":params["background-color"],
                             "height": params["height"],
                          "margin-right":"0px",
                          "border-radius": params["border-radius"]})




# In[10]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {"background-color":"#161616",
         "color":"#ffffff"}



########## HEADER
header = html.Div([dcc.Markdown("**OPTIMAL REBALANCING METHODS**", style={"font-size":"35px"})],

                  style={"padding": "50px",
                        "background":"#161616",
                        "margin":"-15px 0px -0px -10px",
                        "textAlign":"center",
                        "color":"#FFFFFF"})

########## BANDEAU 1
lpp = html.Div([dcc.Markdown("**BENCHMARK**", style={"color":colors["color"]}),
                dcc.Dropdown(id="actifs",
                     options=benchs,
                    value="lpp200025",
                     multi=False)], style={"margin":"0px 0px 0px 0px", "width":"50%",
                                           "padding":"30px 0px 0px 0px",'marginLeft' : '50px'})

input_value = html.Div([dcc.Markdown("**VALUE FOR REBALANCING (see the documentation)**", style={"color":colors["color"],"margin":"0px 0px 0px 50px"}),
                        dcc.Input(id="value", type="text", placeholder="Value for rebalancing", style={"margin":"0px 0px 0px 50px"})])

inputs = html.Div([dcc.Markdown("**VALUE FOR REBALANCING (see the documentation)**", style={"color":colors["color"]}),
                  lpp, input_value], style={"columnCount":2,"margin":"0px 0px 0px 50px"})

methods = html.Div([dcc.Markdown("**METHOD OF REBALANCING**", style={}),
                    dcc.RadioItems(id="Optimizor",options=[{'label': "No rebalancing", 'value': "NR"},
                                            {'label': "Fixed interval rebalancing", 'value': "FIR"},
                                            {'label': "Absolute deviation", 'value': "AD"},
                                            {'label': "Relative deviation", 'value': "RD"},
                                            {'label': "Tracking Error deviation", 'value': "TE"},
                                            {'label': "Standard deviation deviation", 'value': "SDD"},
                                            {'label': "Momentum strategy (TAA)", 'value': "MOM"}],value="NR")],
                   style={"color":"#ffffff","margin":"50px 0px 50px 50px"})

spaces = html.Div([dcc.Markdown("", style={"height":"100px"})])


review = html.Div([te, cost, ret_pf, ret_idx,exp], style={"columnCount":5, "background-color":"#161616",
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
              Output("exp", "children"),
              Input("value", "value"),
              Input("Optimizor", "value"),
             Input("actifs","value"))

def affichage(value, method, lpp_value):
    print(value, method)
    if lpp_value == "lpp200025":
        database, weight, bench, columns = lpp2000, weight_LPP2000_25, "LPP 25", ["Swiss Bond", "Euro Bonds","Mondial Bonds", "Swiss Stocks", "Mondial Stocks"]
        exp="30%"
        resume = resume_2000_25
    elif lpp_value == "lpp200040":
        database, weight, bench, columns = lpp2000, weight_LPP2000_40, "LPP 40", ["Swiss Bond", "Euro Bonds","Mondial Bonds", "Swiss Stocks", "Mondial Stocks"]
        exp="40%"
        resume = resume_2000_40
    else:
        database, weight, bench, columns = lpp2000, weight_LPP2000_60, "LPP 60", ["Swiss Bond", "Euro Bonds","Mondial Bonds", "Swiss Stocks", "Mondial Stocks"]
        exp="55%"
        resume = resume_2000_60
    if value == "":
        value = 0
    if value==None:
        value = 0

    reb = Rebalancing(database, bench, columns, weight, cost=0.0013)
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

    elif method=="SDD":
        reb.standard_deviation_deviation(float(value))

    else:
        reb.momentum(float(value))


    reb.visualisation()
    vis = Visualization(reb.portfolio, reb.database[reb.benchmark], reb.weight)
    cum = vis.fig_cumulative_returns()
    weight = vis.fig_weight_portfolio()
    ret = vis.yearly_return_comparaison()
    tot_cost = np.round(reb.total_cost,3)
    cost = "{}%".format(tot_cost)
    #cost = f"{'%.3f' % reb.total_cost}%"
    tot_te = np.round(reb.te,3)
    te = "{}%".format(tot_te)
    #te = f"{'%.3f' % reb.te}%"


    sunburst_2015_40 = px.sunburst(resume, path=['Sector', 'Indices'], values="Weight %")
    sunburst_2015_40.update_layout(template="plotly_dark", font={"color":"white"},title="Benchmark", title_x=0.5,
                paper_bgcolor="#131313",
                plot_bgcolor="#131313",
                      legend=dict(
                      x=0.03,
                      y=0.97,
                      traceorder='normal',
                      font=dict(size=12)))

    #pf_ret = f"{'%.3f' % reb.annual_pf}%"
    tot_ret = np.round(reb.annual_pf,3)
    pf_ret = "{}%".format(tot_ret)
    #pf_idx = f"{'%.3f' % reb.annual_idx}%"
    tot_idx = np.round(reb.annual_idx,3)
    pf_idx = "{}%".format(tot_idx)
    return cum, weight, sunburst_2015_40, ret, cost,te,pf_ret, pf_idx,exp

if __name__ == '__main__':
    app.run_server(debug=False)