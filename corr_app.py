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

from df_load_func import df_load 


def show_corr():
    # 기본으로 df 출력하기, 인자를 안넘기면 다 반환한다
    df = df_load()
    st.text('당뇨병 차트')
    st.dataframe(df)

    st.subheader('컬럼별로 상관관계를 분석 합니다.')

    multi_list = []
    for column in df.columns:
        multi_list.append(column)
    
    selectionOfUser = st.multiselect('컬럼을 선택 하세요', multi_list)
    
    # 숫자로만 되어있는 df 표시
    st.dataframe(df)

    # 선택을 2개 이상할 때 실행 
    selection_nbr = len(selectionOfUser)
    if selection_nbr >= 2:
        if selection_nbr == 2:
            #print(selectionOfUser)
            # 선택한 리스트의 각각 x, y을 저장
            str_x = selectionOfUser[0]
            str_y = selectionOfUser[1]

            st.success(str_x +'와 ' + str_y + '선택하셨습니다.')
            fig = plt.figure() 
            sb.regplot(data= df, x = str_x, y = str_y)
            plt.xlabel(str_x)
            plt.ylabel(str_y)
            plt.title(str_x + ' 와 ' + str_y + ' 상관 관계 분석')
            st.pyplot(fig)
        
        # 선택 3개 이상일 경우
        if selection_nbr >= 3 :
            st.success(str(selection_nbr) + '컬럼 이상은 FairPlot으로 분석 합니다.')
            column_list = []
            for column in selectionOfUser:
                column_list.append(column)

            #print(column_list)
            df_column = df[ column_list ]
            print(column_list)
            
            #pairplot 은 변수에 저장한 후에 st.pyplot()으로 해야지 됨
            # 다른 차트 하듯이 하면 표시가 안됨
            
            pairplot = sb.pairplot(data = df_column)
            st.pyplot(pairplot)
            
            st.info('차트도 보고 가세요!')
            st.dataframe(df_column.corr())

    elif selection_nbr == 1:
        st.warning('2개 이상 선택을 해주세요')
        