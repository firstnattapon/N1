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

filter = st.sidebar.beta_expander('filter')
filter_0 =  filter.text_input('exchange_id','ftx')
exchange_class = getattr(ccxt, filter_0)
exchange = exchange_class({
    'apiKey': '',
    'secret': '',
    'timeout': 30000,
    'enableRateLimit': True,
})

filter_1 	  =  filter.text_input('filter_1','P')
filter_2 	  =  filter.text_input('filter_2','BULL/USDT')
filter_3 	  =  filter.text_input('filter_3','BEAR/USDT')
filter_4 	  =  filter.text_input('filter_4','DOWN/USDT')
filter_5 	  =  filter.text_input('filter_5','UP/USDT')
time_z      =  filter.text_input('time_z', '1h')
limit_z     =  filter.number_input('limit_z', 1441)

e = exchange.load_markets()
pair_1   = [i for i in e if i[-1] == filter_1]
pair_1   = [i for i in pair_1 if i[-9:] != filter_2]
pair_1   = [i for i in pair_1 if i[-9:] != filter_3]
pair_1   = [i for i in pair_1 if i[-9:] != filter_4]
pair_1   = [i for i in pair_1 if i[-7:] != filter_5]

@st.cache(suppress_st_warning=True)
def z (coin):
  global limit_z ; global time_z
  ohlcv =  exchange.fetch_ohlcv( coin  , time_z , limit=limit_z)
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]
  return Prop

def sum_all_z (Prop):
  pct = pd.DataFrame()
  pct['y'] = Prop.y.pct_change()
  pct['ohlc'] = Prop.y
  pct['cf_buy'] =  pct.y.map( lambda  x : np.where (x > 0 , x  , 0 ))  
  pct['sum_buy'] = pct.cf_buy.cumsum()    
  pct['cf_sell'] =  pct.y.map( lambda  x : np.where (x < 0 , abs(x)  , 0) )  
  pct['sum_sell'] = pct.cf_sell.cumsum() 
  pct['cf_all'] =  pct.y.map( lambda  x : abs(x) )  
  pct['sum_all'] = pct.cf_all.cumsum() 
  pct = pct[['sum_buy', 'sum_sell' ,'sum_all']]
  pct = pct.tail(1)
  pct = pct.reset_index()
  return pct

def A (lp):
  global shift_d ;   global coin ; global limit_a
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit= limit_a )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]
  Prop = Prop[:lp]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop , forecast

def B (lp):
  global shift_d ;   global coin ; global limit_b
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=limit_b )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]
  Prop = Prop[:lp]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop , forecast
  
def C (lp):
  global shift_d ;   global coin ;  global limit_c
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=limit_c )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]
  Prop = Prop[:lp]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop , forecast
  
def D (lp):
  global shift_d ;   global coin ; global limit_c
  ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=limit_c )
  ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
  df =  pd.DataFrame(ohlcv)
  df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

  shift_d = shift_d
  Prop = df
  Prop['ds'] = Prop['t'] 
  Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
  Prop = Prop.iloc[ : , -2:]
  Prop = Prop[:lp]

  m = Prophet( n_changepoints = n_changepoints )
  m.fit(Prop) 
  future = m.make_future_dataframe(periods=shift_d)
  forecast = m.predict(future)
  fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
  st.pyplot() ; #st.write(Prop.tail(1))
  return Prop , forecast
  
def sum_all (Prop ,forecast):
  pct = pd.DataFrame()
  pct['y'] = Prop.y.pct_change()
  pct['ohlc'] = Prop.y
  pct['yhat'] = forecast.yhat
  pct['%'] = ((pct['ohlc'] / pct['yhat']) - 1)*100
  pct['cf_buy'] =  pct.y.map( lambda  x : np.where (x > 0 , x  , 0 ))  
  pct['sum_buy'] = pct.cf_buy.cumsum()    
  pct['cf_sell'] =  pct.y.map( lambda  x : np.where (x < 0 , abs(x)  , 0) )  
  pct['sum_sell'] = pct.cf_sell.cumsum() 
  pct['cf_all'] =  pct.y.map( lambda  x : abs(x) )  
  pct['sum_all'] = pct.cf_all.cumsum() 
  pct = pct[['sum_buy', 'sum_sell' ,'sum_all' , '%' ]]
  st.write(pct.tail(1))  
  
_ , col0 , _  = st.beta_columns(3)
col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)
w , _  = st.beta_columns(2)

Prop = z(pair_1[:1][-1])
df_1 = sum_all_z(Prop)
df_1['index_coin'] = 'BTC/USDT'
for i in pair_1[1:]:
  Prop = z(i)
  df_2 = sum_all_z(Prop)
  df_2['index_coin'] = i
  df_1 = pd.concat([df_1, df_2], axis=0 , ignore_index=True)
df_1 =  df_1.sort_values(['sum_all'] , axis=0 ,ascending=False)

df_1 = df_1[df_1['index'] >= (limit_z-1)] 
st.sidebar.write(df_1)  

sort 	  =  st.sidebar.number_input('sort',value=15)
df_f = df_1.head(sort)
df_f = df_f.index_coin
pair_2 = df_f

coin_beta_expander = st.sidebar.beta_expander('coin')
coin = coin_beta_expander.radio('coin', [ i for i  in enumerate(pair_2)] ) ; coin = coin[-1]
timeframe =  st.sidebar.selectbox('time',('1d' , '15m' ,'1h' , '4h'))
limit_a     =   st.sidebar.number_input('limit_a',value=90)
limit_b     =   st.sidebar.number_input('limit_b',value=180)
limit_c     =   st.sidebar.number_input('limit_c',value=270)
limit_d     =   st.sidebar.number_input('limit_d',value=365)
n_changepoints =  st.sidebar.number_input('n_changepoints',min_value=0,value=25,step=1)
shift_d   = st.sidebar.number_input('shift_d', 1)  

with col0:
  lb =   st.number_input('Looking_back',value=-1)

with col1:
  Prop , forecast = A(lb)
  col1_expander = st.beta_expander('90' , expanded=True)
  with col1_expander:  
    sum_all(Prop , forecast)

with col2:
  Prop , forecast = B(lb)
  col2_expander = st.beta_expander('180' , expanded=True)
  with col2_expander:  
    sum_all(Prop , forecast)
  
with col3:
  Prop , forecast = C(lb)
  col3_expander = st.beta_expander('270' , expanded=True)
  with col3_expander:  
    sum_all(Prop , forecast)
    
with col4:
  Prop , forecast = D(lb) 
  col4_expander = st.beta_expander('365' , expanded=True)
  with col4_expander:  
    sum_all(Prop , forecast)
    
with w:
  vae = st.beta_expander('vae')
  with vae:  
    wr = """
    pair_0 ['ALGO/USDT', 'BAL/USDT', 'CHZ/USDT', 'KNC/USDT', 'MATIC/USDT', 'PAXG/USDT', 'XRP/USDT', 'XTZ/USDT']\n
    pair_1 ['DOGE/USDT', 'DOT/USDT', 'EOS/USDT', 'OMG/USDT', 'THETA/USDT', 'TRX/USDT', 'ZEC/USDT']\n
    pair_2 ['ADA/USDT', 'ATOM/USDT', 'BNB/USDT', 'ETC/USDT', 'MKR/USDT', 'RUNE/USDT']\n
    pair_3 ['COMP/USDT', 'NEO/USDT', 'SUSHI/USDT', 'UNI/USDT', 'VET/USDT', 'YFI/USDT']\n
    pair_4 ['BCH/USDT', 'BTC/USDT', 'ETH/USDT', 'LINK/USDT', 'LTC/USDT']\n
    pair_5 ['SOL/USDT', 'SXP/USDT', 'TOMO/USDT']\n
    """
    st.write(wr)  
    
