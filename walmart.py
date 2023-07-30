# Lazım olan kitabxanalar
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
import pickle
import time
import PIL

# Potensial xəbərdarlıqların filterlənməsi
warnings.filterwarnings(action = 'ignore')

# Datasetin yüklənməsi
df = pd.read_csv('cleaned.data.csv')
    

# İlkin əməliyyatlardan ibarət sinifin yaradılması
class InitialPreprocessor(BaseEstimator, TransformerMixin):
    # fit funksiyasının yaradılması
    def fit(self, X, y = None):
        # Nominal dəyişənlərdən ibarət list data strukturunun yaradılması
        self.nominal_features = ['Type']
        
        # Bulean dəyişənlərdən ibarət list data strukturunun yaradılması
        self.boolean_features = ['Isholiday']
        
        # Listlərin geri qaytarılması
        return self
    
    # transform funksiyasının yaradılması
    def transform(self, X, y = None):
        # Nominal dəyişənlərdə ola biləcək potensial boşluqların silinib bütün dəyərlərin ilk hərflərinin böyüdülməsi
        X[self.nominal_features] = X[self.nominal_features].applymap(func = lambda x: x.strip().capitalize(), na_action = 'ignore')
        
        # Bulean dəyişənlərdə olan dəyərlərin integer data növünə dəyişdirilməsi
        X[self.boolean_features] = X[self.boolean_features].applymap(func = lambda x: int(x))
        
        # Əməliyyat tətbiq olunmuş asılı olmayan dəyişənlərin geri qaytarılması
        return X


# Şəkilin yüklənməsi
walmart_image =Image.open(fp="walmart.jfif")

    
# Əsas səhifənin yaradılması
interface = st.container()

# Modelin yüklənməsi
with open(file = 'model.pkl', mode = 'rb') as pickled_model:
    model = pickle.load(file = pickled_model)

# Generate the date range
start_date = '2010-02-05'
end_date = '2012-10-26'
date_range = pd.date_range(start=start_date, end=end_date)
dates = date_range.strftime('%Y-%m-%d').tolist()
# Əsas səhifənin yaradılması
interface = st.container()

# Yan səhifənin yaradılması
sidebar = st.sidebar.container()

# Əsas səhifənin göstərilməsi
with interface:
    # Səhifənin adının göstərilməsi
    st.title(body = 'Walmart Sales Prediction')
    
    # Səhifədə titanik şəklinin göstərilməsi
    st.image(image = walmart_image)
    
    # Səhifədə başlığın göstərilməsi
    st.header(body = 'Project Description')
    
    # Səhifədə proyektlə bağlı məlumatın verilməsi
    st.text(body = """
    The purpose of this case study is to build a sales forecasting model that can accurately predict future sales for Walmart       stores. This model will help Walmart optimize its inventory levels, plan promotions, and increase profits by ensuring that     they always have the right amount of stock to meet customer demand.
    """)
    
    
    # Səhifədə IsHoliday sütun adın yaradılması
    IsHoliday, Year, Type, Month = st.columns(spec = [1, 1, 1, 1])


    # Səhifədə holiday sütunu üçün dəyərlərin təyin olunması
    with IsHoliday:
        IsHoliday = st.radio(label = 'Is the week a special holiday week?', options = [True, False], horizontal = True)
    
    # Səhifədə year sütunu üçün dəyərlərin təyin olunması
    with Year:
        Year = st.selectbox(label='Select Year', options=['2010', '2011', '2012'])
        
    
    # Səhifədə date sütunu üçün dəyərlərin təyin olunması      
    #date = st.selectbox(label='Select Date', options=dates)
    date = st.empty()
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə temperature sütunu üçün dəyərlərin təyin olunması
    temperature = st.slider(label = 'Temperature', min_value = 1, max_value = 101, value = int(df.Temperature.mean()))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə store sütunu üçün dəyərlərin təyin olunması
    store = st.slider(label = 'Store', min_value = 1, max_value = 45, value = int(df.Store.mean()))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə day sütunu üçün dəyərlərin təyin olunması
    day = st.slider(label = 'Day', min_value = 1, max_value = 7, value = 1)
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə fuel_price sütunu üçün dəyərlərin təyin olunması
    fuel_price = st.text_input(label='Fuel price', value=str(int(df.Fuel_price.mean())))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə cpi sütunu üçün dəyərlərin təyin olunması
    cpi = st.slider(label = 'CPI', min_value = 126, max_value = 133, value = int(df.Cpi.mean()))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə size sütunu üçün dəyərlərin təyin olunması
    size = st.slider(label = 'Size', min_value = 34875, max_value = 219622, value = int(df.Size.mean()))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə department sütunu üçün dəyərlərin təyin olunması
    department = st.slider(label = 'Department', min_value = 1, max_value = 99, value = int(df.Dept.mean()))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    
    # Səhifədə unemployment sütunu üçün dəyərlərin təyin olunması
    unemployment = st.text_input(label='Unemployment rate', value=str(int(df.Unemployment.mean())))
    
    # Səhifənin uzun xətt ilə bölünməsi
    st.markdown(body = '***')
    

    # Səhifədə type sütunu üçün dəyərlərin təyin olunması
    with Type:
        Type = st.selectbox(label = 'Select Type', options = ['A', 'B', 'C'])
        
    # Səhifədə month sütunu üçün dəyərlərin təyin olunması
    with Month:
        Month = st.selectbox(label = 'Select Month', options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']) 
    
    # Lüğət data strukturunda asılı olmayan dəyişənlərin saxlanılması
    data_dictionary = {
                   'Store': store,
                   'Dept': department,
                   'Date': date,
                   'Day': day,
                   'Month':Month,
                   'Year':Year,
                   'Isholiday': IsHoliday,
                   'Type': Type,
                   'Temperature': temperature,
                   'Fuel_price': fuel_price,
                   'Cpi':cpi,
                   'Size':size,
                   'Unemployment':unemployment} 
    
    # Lüğət data strukturunun Pandas dataframe data strukturuna çevirilməsi
    input_df = pd.DataFrame(data=data_dictionary, index=[0])
       
    
    # Səhifədə kiçik başlığın göstərilməsi
    st.subheader(body = 'Model Prediction')
    
    
    # Səhifədə düymənin yaradılması
    if st.button('Predict'):
        # Perform prediction using the model
        predicted_value = model.predict(X=input_df)
    
        # Proqnoz zamanı döngünün yaradılması
        with st.spinner(text='Sending input features to model...'):
            # Döngü bitəndən sonra iki saniyəlik pauza verilməsi
            time.sleep(2)

            # Pauzadan dərhal sonra müvəffəqiyyət ismarıcının göstərilməsi
            st.success('Your prediction is ready!')

            # İsmarıcdan sonra bir saniyəlik pauza verilməsi
            time.sleep(1)

            # Modelin proqnoz etdiyi qiymətin göstərilməsi
            st.markdown(f'Model output: Predicted value is **{predicted_value[0]:.2f}**')

# Yan səhifənin göstərilməsi
with sidebar:
    # Səhifənin adının göstərilməsi
    st.title('Data Dictionary')
    
    # Səhifədə asılı olmayan dəyişənlərin izahlarının göstərilməsi
    st.markdown('''
    - **Store**: The unique store number (numeric)
    - **Dept**: The unique department number (numeric)
    - **Date**: Date of sales (date)
    - **Month**: Month of sales (date)
    - **IsHoliday**: Whether the week is a special holiday week (categorical: ‘FALSE’, ‘TRUE’)
    - **Weekly_Sales**: Target variable - weekly sales (numeric)
    - **Type**: The type of store (categorical: 'A', 'B', 'C')
    - **Size**: The size of the store in square feet (numeric)
    - **Temperature**: Average temperature in the region (numeric)
    - **Fuel_Price**: Cost of fuel in the region (numeric)
    - **CPI**: Consumer Price Index (numeric)
    - **Unemployment**: Unemployment rate (numeric)
    ''')
