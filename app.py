import ccxt
import datetime  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
pd.set_option("display.precision", 6)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.beta_set_page_config(
  page_title="App",
  layout="wide",
  initial_sidebar_state="expanded")

# sns.set_style("whitegrid")

exchange = ccxt.binance({'apiKey': ''   ,'secret':  ''  , 'enableRateLimit': True }) 
e = exchange.load_markets()

filter 	  =  st.sidebar.text_input('filter','T')
pair 		   = [i for i in e if i[-1] == filter]

coin_beta_expander = st.sidebar.beta_expander('coin')
coin = coin_beta_expander.radio('coin',tuple(pair))
  
timeframe =  st.sidebar.selectbox('time',('1d' , '15m' ,'1h' , '4h'))
# limit     =   st.sidebar.selectbox('limit',(180 , 270 , 365))
n_changepoints =  st.sidebar.number_input('n_changepoints',min_value=0,value=25,step=1)
shift_d   = st.sidebar.number_input('shift_d', 1)

def A ():
  global shift_d
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=90 )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop , forecast

def B ():
  global shift_d
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=180 )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop
  
def C ():
  global shift_d
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=270 )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1)) 
  return Prop  
  
def D ():
  global shift_d
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=365 )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop
  
def sum_all (Prop ,forecast ):
  pct = pd.DataFrame()
  pct['y'] = Prop.y.pct_change()
  pct['ohlc'] = Prop.y
  pct['yhat'] = forecast.yhat
  pct['%'] = pct['ohlc'] / pct['yhat']
  pct['cf_buy'] =  pct.y.map( lambda  x : np.where (x > 0 , x  , 0 ))  
  pct['sum_buy'] = pct.cf_buy.cumsum()    
  pct['cf_sell'] =  pct.y.map( lambda  x : np.where (x < 0 , abs(x)  , 0) )  
  pct['sum_sell'] = pct.cf_sell.cumsum() 
  pct['cf_all'] =  pct.y.map( lambda  x : abs(x) )  
  pct['sum_all'] = pct.cf_all.cumsum() 
  pct = pct[['sum_buy', 'sum_sell' ,'sum_all' , 'ohlc' , 'yhat' , '%' ]]
  st.write(pct.tail(1))  
  
col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)

with col1:
  Prop , forecast = A()
  col1_expander = st.beta_expander('90' , expanded=True)
  with col1_expander:  
    sum_all(Prop , forecast)

with col2:
  Prop = B()
  col2_expander = st.beta_expander('180' , expanded=True)
  with col2_expander:  
    sum_all(Prop)
  
with col3:
  Prop = C()
  col3_expander = st.beta_expander('270' , expanded=True)
  with col3_expander:  
    sum_all(Prop)  
    
with col4:
  Prop = D() 
  col4_expander = st.beta_expander('365' , expanded=True)
  with col4_expander:  
    sum_all(Prop)  
  
#   f, (ax1, ax2) = plt.subplots(2  , figsize=(15,15) )
#   ax1.plot(pct.sum_all)
#   ax2.plot(pct.sum_buy)
#   ax2.plot(pct.sum_sell)
#   st.pyplot() 


