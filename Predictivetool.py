#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly as plotlyoffline
import plotly.express as px
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils


# In[2]:


# Import data
df_data = pd.read_csv('Mental_Health_CfC_State.csv')
df_data = df_data.fillna(0)
df_data = df_data[df_data['State'] != 'Hawaii']
df_data = df_data[df_data['State'] != 'Maine']
df_data = df_data[df_data['State'] != 'Montana']
df_data = df_data[df_data['State'] != 'North Dakota']
df_data = df_data[df_data['State'] != 'West Virginia']


# In[3]:


state_list = df_data['State'].unique().tolist()
df_final = pd.DataFrame(columns = ['State','Intercept','Coeff_HI','Coeff_Death','Coeff_UC','RMSE','R_squared','Predicted_AnxDep'])
df_final['State'] = state_list


# In[4]:


for i in range(len(df_final)):
    us_state = df_final['State'][i]
    data_state = df_data[df_data['State'] == us_state]
    
    #extract columns with numerical data (exclude response variable)
    df = data_state.iloc[:, [2,4,5]]
    #normalize the dataset
    data_norm = df.sub(df.mean(axis=0), axis=1)/df.mean(axis=0)
    #add the response variable(and non-numeric variables removed above)
    data_norm['Hispanic_AnxDep'] = data_state['Hispanic_AnxDep']

    #data_norm2 = (data-data.min())/(data.max()-data.min())

    #Building the model
    df_1 = data_norm.values
    #X is the set of independent variables
    X = df_1[:,0:3]
    #Y is the dependent variable
    Y = df_1[:,3]
    #split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

    #Build the model
    Model = linear_model.LinearRegression()
    Model.fit(X_train, Y_train)
    y_pred = Model.predict(X_test)

    # Root Mean Squared Deviation
    rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))      
    corr_matrix = np.corrcoef(Y_test, y_pred)
    r2_value = (corr_matrix[0,1])**2
    y_tplus1 = Model.intercept_ + Model.coef_[0]*X[-1,0] + Model.coef_[1]*X[-1,1] + Model.coef_[2]*X[-1,2]
    
    df_final['Intercept'][i] = Model.intercept_
    df_final['Coeff_HI'][i] = Model.coef_[0]
    df_final['Coeff_Death'][i] = Model.coef_[1]
    df_final['Coeff_UC'][i] = Model.coef_[2]
    df_final['RMSE'][i] = rmsd
    df_final['R_squared'][i] = r2_value
    df_final['Predicted_AnxDep'][i] = y_tplus1


# In[5]:


# Create traces
fig_Rs = go.Figure()
fig_Rs.add_trace(go.Scatter(x=df_final['State'], y=df_final['R_squared'],
                    mode='lines',
                    name='lines'))

# Edit the layout
fig_Rs.update_layout(
            template = "ggplot2",
            title = 'R-squared values achieved across states',
            xaxis_title='States',
            yaxis_title='R-squared values',
            autosize=False,
            width=1200,
            height=600,
            showlegend = False,
            margin ={'l':0,'t':0,'b':0,'r':0},
            hovermode='closest',
        )

fig_Rs.add_annotation(dict(font=dict(color='black',size=16),
                                        x=0.01,
                                        y=1,
                                        showarrow=False,
                                        text='R-squared (Interactive)',
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))


plotlyoffline.offline.plot(fig_Rs, filename = 'R-squaredvalues_states.html', 
                         include_plotlyjs='https://cdn.plot.ly/plotly-1.42.3.min.js') #offline version


# In[6]:


# Create traces
fig_RMSE = go.Figure()
fig_RMSE.add_trace(go.Scatter(x=df_final['State'], y=df_final['RMSE'],
                    mode='lines',
                    name='lines'))

# Edit the layout
fig_RMSE.update_layout(
            template = "plotly_white",
            title = 'RMSE values achieved across states',
            xaxis_title='States',
            yaxis_title='RMSE',
            autosize=False,
            width=1200,
            height=600,
            showlegend = False,
            margin ={'l':0,'t':0,'b':0,'r':0},
            hovermode='closest',
        )

fig_RMSE.add_annotation(dict(font=dict(color='black',size=16),
                                        x=0.01,
                                        y=1,
                                        showarrow=False,
                                        text='RMSE (Interactive)',
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))


plotlyoffline.offline.plot(fig_RMSE, filename = 'RMSE_states.html', 
                         include_plotlyjs='https://cdn.plot.ly/plotly-1.42.3.min.js') #offline version


# In[7]:


# Create traces
fig_Pred = go.Figure()
fig_Pred.add_trace(go.Scatter(x=df_final['State'], y=df_final['Predicted_AnxDep'],
                    mode='lines',
                    name='lines'))

# Edit the layout
fig_Pred.update_layout(
            template = "plotly_white",
            title = 'Predicted percentage of adults showing symptoms of anxiety and disorders in next 2-weeks',
            xaxis_title='States',
            yaxis_title='Predicted anxiety and disorder level',
            autosize=False,
            width=1200,
            height=600,
            showlegend = False,
            margin ={'l':0,'t':0,'b':0,'r':0},
            hovermode='closest',
        )

fig_Pred.add_annotation(dict(font=dict(color='black',size=12),
                                        x=0.01,
                                        y=1,
                                        showarrow=False,
                                        text='Predicted percentage of adults showing symptoms of anxiety and disorders in next 2-weeks (Interactive)',
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))


plotlyoffline.offline.plot(fig_Pred, filename = 'Prediction_states.html', 
                         include_plotlyjs='https://cdn.plot.ly/plotly-1.42.3.min.js') #offline version

