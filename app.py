import math
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
from sklearn.impute import SimpleImputer
import plotly.express as px
import pandas_ta as ta
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
import plotly.graph_objects as go
from datetime import date
from keras.models import load_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# from sklearn.externals import joblib
from datetime import datetime, timedelta
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

st.set_page_config(layout="wide")

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Enter Stock Symbol', 'HDFCBANK.NS')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2011-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2021-01-15'))

data = yf.download(ticker, start=start_date, end=end_date)

sia = SentimentIntensityAnalyzer()

# tabs = st.sidebar.radio("View", ("Price Chart", "Fundamental Data", "Technical Analysis", "News"))

pricing_data,fundamental_data,news,HDFC_STOCK_ANALYSIS = st.tabs(["Pricing Data","Fundamental Data", "Top 10 News","HDFC STOCK ANALYSIS"])
# Price chart
with pricing_data:
    st.subheader('Pricing Data')
    fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Stock Price')
    st.plotly_chart(fig)
    st.write(data)
    data2 = data
    data2['% Change'] = data['Adj Close']/data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is ',annual_return,'%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ',stdev*100,'%')
    st.write('Risk Adj. Return is ',annual_return/(stdev*100))

    # histogram
    st.subheader('Daily Returns Histogram')
    fig_hist = px.histogram(data2, x='% Change', nbins=50)
    st.plotly_chart(fig_hist)

# Fundamental data
with fundamental_data:
    st.subheader('Fundamental Data')

    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list = True)
    technical_indicator = st.selectbox('Tech Indicator',options=ind_list)
    method = technical_indicator
    data = yf.download(ticker, start=start_date, end=end_date)
    indicator = pd.DataFrame(getattr(ta,method)(low=data['Low'],close=data['Close'],high=data['High'],open=data['Open'],volume=data['Volume']))
    indicator['Close'] = data['Close']
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)

    key = 'VH3DJQSPYZMVDIAN'
    # VH3DJQSPYZMVDIAN
    # 1A22AEJGEDW3O4YI
    try:
            key = '1A22AEJGEDW3O4YI'
            fd = FundamentalData(key)
            balance_sheet, _ = fd.get_balance_sheet_annual(symbol=ticker)
            income_statement, _ = fd.get_income_statement_annual(symbol=ticker)
            cash_flow, _ = fd.get_cash_flow_annual(symbol=ticker)

            st.subheader('Balance Sheet')
            st.write(balance_sheet)

            st.subheader('Income Statement')
            st.write(income_statement)

            st.subheader('Cash Flow Statement')
            st.write(cash_flow)
    except Exception as e:
        st.write("Sorry, fundamental data is not available for this stock.")



# News
with news:
    st.subheader('Latest News')
    fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Stock Price')
    st.plotly_chart(fig)
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(min(10, len(df_news))):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])

        title_sentiment = sia.polarity_scores(df_news['title'][i])['compound']
        st.write(f'Title Sentiment: {title_sentiment}')

        news_sentiment = sia.polarity_scores(df_news['summary'][i])['compound']
        st.write(f'News Sentiment: {news_sentiment}')





# Technical Analysis
with HDFC_STOCK_ANALYSIS:
    st.subheader('HDFC STOCK ANALYSIS')
    
    # Load the pre-trained models
    
    model_lr = joblib.load('lr_model.joblib')
    model_dt = joblib.load('tree_model.joblib')
    model_rf = joblib.load('rf_model.joblib')
    model_xgb = joblib.load('xgb_model.joblib')
    model_svm = joblib.load('svr_model.joblib')
    model_knn = joblib.load('knn_model.joblib')
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
    gru_model = tf.keras.models.load_model('gru_model.h5')

    ticker = 'HDFCBANK.NS'
    df = yf.download(ticker, start='1996-01-01', end='2024-01-01')

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset)* .8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)


    test_data = scaled_data[training_data_len-60: , :]
    x_test = []
    y_test = dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    # LSTM Predictions
    lstm_predictions = lstm_model.predict(x_test)
    lstm_predictions_inv = scaler.inverse_transform(lstm_predictions)
    gru_predictions = gru_model.predict(x_test)
    gru_predictions_inv = scaler.inverse_transform(gru_predictions)

    # Select start and end dates
    start_date_custom = pd.Timestamp(st.date_input('From Date', min_value=pd.Timestamp('1950-01-01'), max_value=pd.Timestamp.today(), value=start_date))
    end_date_custom = pd.Timestamp(st.date_input('To Date', min_value=pd.Timestamp('1950-01-01'), max_value=pd.Timestamp.today(), value=end_date))

    # Check if start_date_custom is before 1950
    if start_date_custom.year < 1950:
        st.warning('Please select a start date after 1950.')
    else:
        # Load HDFC stock data

        hdfc_data = pd.read_csv('B4_A2_2preprocess.csv')
        hdfc_data['Date'] = pd.to_datetime(hdfc_data['Date'], format='%d-%m-%Y')

        # Filter HDFC data for the selected date range
        hdfc_data_filtered = hdfc_data[
            (hdfc_data['Date'] >= start_date_custom) & (hdfc_data['Date'] <= end_date_custom)
        ]
        hdfc_data_filtered = hdfc_data_filtered.dropna()

        if not hdfc_data_filtered.empty:
            fig_hdfc_actual = px.line(hdfc_data_filtered, x='Date', y='Close', title='Actual Close Price of HDFCBANK.NS')
            st.plotly_chart(fig_hdfc_actual, use_container_width=True)
            # Prepare the data for prediction
            X_custom = hdfc_data_filtered[['Open', 'High', 'Low', 'AdjClose', 'Volume']]
            y_custom_actual = hdfc_data_filtered['Close']

            # Make predictions using linear regression model
            y_custom_lr_pred = model_lr.predict(X_custom)

            scaler = StandardScaler()
            X_custom_scaled = scaler.fit_transform(X_custom)


            # Make predictions using decision tree model
            y_custom_dt_pred = model_dt.predict(X_custom)


            # Make predictions using random forest model
            y_custom_rf_pred = model_rf.predict(X_custom)
            # Make predictions using XGBoost model
            y_custom_xgb_pred = model_xgb.predict(X_custom)

            # Make predictions using SVM model
            y_custom_svm_pred = model_svm.predict(X_custom[['AdjClose']])

            # Make predictions using KNN model
            y_custom_knn_pred = model_knn.predict(X_custom[['AdjClose']])


            # Create a DataFrame with actual and predicted values
            custom_analysis_data = pd.DataFrame({
                'Date': hdfc_data_filtered['Date'],
                'Actual Close': y_custom_actual,
                'XGBoost Prediction': y_custom_xgb_pred,
                'Linear Regression Prediction': y_custom_lr_pred,
                'Decision Tree Prediction': y_custom_dt_pred,
                'Random Forest Prediction': y_custom_rf_pred,
                'SVM Prediction': y_custom_svm_pred,
                'KNN Prediction': y_custom_knn_pred
                

                # 'SVM3 Prediction': y_custom_svm3_pred
            })
            # Create a DataFrame for LSTM predictions
            train = data[:training_data_len]
            valid = data[training_data_len:]
            lstm_predictions_df = pd.DataFrame(lstm_predictions_inv, columns=['LSTM Predictions'])
            gru_predictions_df = pd.DataFrame(gru_predictions_inv, columns=['GRU Predictions'])
            valid['LSTM Predictions'] = lstm_predictions_inv
            valid['GRU Predictions'] = gru_predictions_inv
            # Display the data frame
            st.write(custom_analysis_data)
            st.write("LSTM & GRU Predictions:")
            st.dataframe(valid)
            # Calculate daily percentage change
            hdfc_data_filtered['% Change'] = hdfc_data_filtered['AdjClose'].pct_change()

            # Calculate annual return
            annual_return = hdfc_data_filtered['% Change'].mean() * 252 * 100
            st.write('Annual Return is ', annual_return, '%')

            # Calculate standard deviation
            stdev = np.std(hdfc_data_filtered['% Change']) * np.sqrt(252)
            st.write('Standard Deviation is ', stdev * 100, '%')

            # Calculate risk-adjusted return
            risk_adj_return = annual_return / (stdev * 100)
            st.write('Risk Adj. Return is ', risk_adj_return)

            # Plot time vs actual and predicted close
            fig_custom = px.line(custom_analysis_data, x='Date', y=custom_analysis_data.columns[1:], title='Predictions Comparison')
            st.plotly_chart(fig_custom, use_container_width=True, width=1200, height=800)

            # Convert LSTM predictions to a DataFrame
            lstm_predictions_df = pd.DataFrame({
                'Date': df.index[training_data_len:],
                'Actual Close Price': df['Close'][training_data_len:],
                'LSTM Predictions': lstm_predictions_inv.ravel()
            })

            # Set 'Date' as the index
            lstm_predictions_df.set_index('Date', inplace=True)

            # Plot LSTM predictions using Streamlit
            st.line_chart(lstm_predictions_df)

            
            # Convert LSTM predictions to a DataFrame
            gru_predictions_df = pd.DataFrame({
                'Date': df.index[training_data_len:],
                'Actual Close Price': df['Close'][training_data_len:],
                'GRU Predictions': gru_predictions_inv.ravel()
            })

            # Set 'Date' as the index
            gru_predictions_df.set_index('Date', inplace=True)

            # Plot LSTM predictions using Streamlit
            st.line_chart(gru_predictions_df)

            with st.form(key='prediction_form'):
                    open_val = st.number_input('Open', value=0.0)
                    high_val = st.number_input('High', value=0.0)
                    low_val = st.number_input('Low', value=0.0)
                    adjclose_val = st.number_input('AdjClose', value=0.0)
                    volume_val = st.number_input('Volume', value=0.0)

                    if st.form_submit_button('Predict'):
                        X_custom_input_svm = np.array([[adjclose_val]])  # Keep only the 'AdjClose' feature for SVM
                        X_custom_input_knn = np.array([[adjclose_val]])  # Keep only the 'AdjClose' feature for KNN

                        # Make predictions using SVM model
                        y_custom_svm_pred = model_svm.predict(X_custom_input_svm.reshape(-1, 1))

                        # Make predictions using KNN model
                        y_custom_knn_pred = model_knn.predict(X_custom_input_knn.reshape(-1, 1))

                        # For other models, use all features
                        X_custom_input = np.array([[open_val, high_val, low_val, adjclose_val, volume_val]])

                        y_custom_lr_pred = model_lr.predict(X_custom_input)
                        y_custom_dt_pred = model_dt.predict(X_custom_input)
                        y_custom_rf_pred = model_rf.predict(X_custom_input)
                        y_custom_xgb_pred = model_xgb.predict(X_custom_input)

                        # Display the predictions
                        st.write('Linear Regression Prediction:', y_custom_lr_pred[0])
                        st.write('Decision Tree Prediction:', y_custom_dt_pred[0])
                        st.write('Random Forest Prediction:', y_custom_rf_pred[0])
                        st.write('XGBoost Prediction:', y_custom_xgb_pred[0])
                        st.write('SVM Prediction:', y_custom_svm_pred[0])
                        st.write('KNN Prediction:', y_custom_knn_pred[0])
            # Checkbox to show visualizations
            show_visualizations = st.sidebar.checkbox('Show Visualizations', value=True)
            if show_visualizations:
                # Scatter plot
                st.subheader('Scatter Plot of Actual vs Predicted Close')
                fig_scatter = px.scatter(custom_analysis_data, x='Actual Close', y=custom_analysis_data.columns[1:], title='Actual vs Predicted Close')
                st.plotly_chart(fig_scatter)

                # Line plot
                fig_line = go.Figure()
                for column in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
                    fig_line.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=hdfc_data_filtered[column], mode='lines', name=column))
                fig_line.update_layout(title='Stock Price Analysis', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig_line)

                # Candlestick chart
                fig_candlestick = go.Figure(data=[go.Candlestick(x=hdfc_data_filtered['Date'],
                                                                open=hdfc_data_filtered['Open'],
                                                                high=hdfc_data_filtered['High'],
                                                                low=hdfc_data_filtered['Low'],
                                                                close=hdfc_data_filtered['Close'])])
                fig_candlestick.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig_candlestick)

                # Volume analysis
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(x=hdfc_data_filtered['Date'], y=hdfc_data_filtered['Volume'], name='Volume'))
                fig_volume.update_layout(title='Volume Analysis', xaxis_title='Date', yaxis_title='Volume')
                st.plotly_chart(fig_volume)

                # Moving averages
                hdfc_data_filtered['MA50'] = hdfc_data_filtered['Close'].rolling(window=50).mean()
                hdfc_data_filtered['MA200'] = hdfc_data_filtered['Close'].rolling(window=200).mean()
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=hdfc_data_filtered['Close'], mode='lines', name='Close'))
                fig_ma.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=hdfc_data_filtered['MA50'], mode='lines', name='MA50'))
                fig_ma.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=hdfc_data_filtered['MA200'], mode='lines', name='MA200'))
                fig_ma.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig_ma)

                # Relative Strength Index (RSI)
                delta = hdfc_data_filtered['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # Plot RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=rsi, mode='lines', name='RSI', line=dict(color='blue')))
                fig_rsi.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=[70]*len(hdfc_data_filtered), mode='lines', name='Overbought', line=dict(color='red', dash='dash')))
                fig_rsi.add_trace(go.Scatter(x=hdfc_data_filtered['Date'], y=[30]*len(hdfc_data_filtered), mode='lines', name='Oversold', line=dict(color='green', dash='dash')))
                fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI', showlegend=True)
                st.plotly_chart(fig_rsi)


                # Histogram of daily price changes
                price_changes = hdfc_data_filtered['Close'].diff()
                fig_hist_price_changes = px.histogram(x=price_changes, title='Histogram of Daily Price Changes', labels={'x': 'Price Changes'})
                fig_hist_price_changes.update_traces(marker_color='blue', opacity=0.7)
                fig_hist_price_changes.update_layout(xaxis_title='Price Changes', yaxis_title='Count')
                st.plotly_chart(fig_hist_price_changes)


                corr = hdfc_data_filtered[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']].corr()
                fig_heatmap = px.imshow(corr, color_continuous_scale='RdBu', title='Correlation Heatmap')
                st.plotly_chart(fig_heatmap)

                fig_scatter_matrix = px.scatter_matrix(hdfc_data_filtered, dimensions=['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
                st.plotly_chart(fig_scatter_matrix)

                fig_box = px.box(hdfc_data_filtered, y='Close', title='Box Plot of Close Prices')
                st.plotly_chart(fig_box)

                fig_violin = px.violin(hdfc_data_filtered, y='Close', title='Violin Plot of Close Prices')
                st.plotly_chart(fig_violin)

                # Bar chart of average stock price by month
                hdfc_data_filtered['Month'] = hdfc_data_filtered['Date'].dt.month
                avg_price_by_month = hdfc_data_filtered.groupby('Month')['Close'].mean().reset_index()
                fig_bar_monthly_avg_price = px.bar(avg_price_by_month, x='Month', y='Close', title='Average Stock Price by Month', labels={'Month': 'Month', 'Close': 'Average Price'})
                st.plotly_chart(fig_bar_monthly_avg_price)

                # Bar chart of average stock price by year
                hdfc_data_filtered['Year'] = hdfc_data_filtered['Date'].dt.year
                avg_price_by_year = hdfc_data_filtered.groupby('Year')['Close'].mean().reset_index()
                fig_bar_yearly_avg_price = px.bar(avg_price_by_year, x='Year', y='Close', title='Average Stock Price by Year', labels={'Year': 'Year', 'Close': 'Average Price'})
                st.plotly_chart(fig_bar_yearly_avg_price)

                # Bar chart of trading volume by month
                avg_volume_by_month = hdfc_data_filtered.groupby('Month')['Volume'].mean().reset_index()
                fig_bar_monthly_avg_volume = px.bar(avg_volume_by_month, x='Month', y='Volume', title='Average Trading Volume by Month', labels={'Month': 'Month', 'Volume': 'Average Volume'})
                st.plotly_chart(fig_bar_monthly_avg_volume)

                # Bar chart of trading volume by year
                avg_volume_by_year = hdfc_data_filtered.groupby('Year')['Volume'].mean().reset_index()
                fig_bar_yearly_avg_volume = px.bar(avg_volume_by_year, x='Year', y='Volume', title='Average Trading Volume by Year', labels={'Year': 'Year', 'Volume': 'Average Volume'})
                st.plotly_chart(fig_bar_yearly_avg_volume)

                # Compute autocorrelation for a range of lags
                autocorr_values = [hdfc_data_filtered['Close'].autocorr(lag) for lag in range(len(hdfc_data_filtered))]

                # Plot autocorrelation
                fig_autocorrelation = px.line(x=np.arange(len(hdfc_data_filtered)), y=autocorr_values, title='Autocorrelation Plot of Close Prices')
                st.plotly_chart(fig_autocorrelation)


# pip install pipreqs
# pipreqs --encoding=utf8
