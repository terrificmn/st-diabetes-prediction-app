#import
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import h5py
from tensorflow.keras.models import load_model
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

# 주요 함수들 모듈화 임포트 
from func_df_load import run_df_load
from search_app import run_search
from corr_app import show_corr
from ml_app import run_ml_app
from df_load_func import df_load 


st.set_page_config(page_title='Diabetes Prediction', layout='wide', initial_sidebar_state='auto')

def main():
    st.title('당뇨병 예측 AI')
    st.image('image/diabetes_1280.jpg')
    st.info('왼쪽 Menu에서 선택 하세요.')
    select_list =[
                    '원하는 메뉴를 선택하세요', '자료 보기(데이터프레임)', '상관관계 분석', '예측하기'
        ]
    select_choice = st.sidebar.selectbox('Menu', select_list)
    
    
    if select_choice == '선택하세요':
        pass

    if select_choice == '자료 보기(데이터프레임)':
        run_df_load()
    
    elif select_choice == '상관관계 분석':
        show_corr()

    elif select_choice == '예측하기':
        run_ml_app()
        


if __name__ == '__main__':  
    main() # main() 함수 호출


