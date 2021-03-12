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
import re
from PIL import Image # 이미지 처리 라이브러리

from df_load_func import df_load 

# joblib 을 이용해서 저장하기 (스케일러 저장하기)
#joblib.dump(mmX, 'mmX.pkl')

# 충돌 일어날때 가상환경에서 업그레이드 해주기
# pip install scikit-learn-0.23.2 

# 예측 본론

# 학습 => 오차를 줄이는 것
# validataion은 에포크 끝난 (학습 1회 끝남) 문제를 주고, 계산만 함
# 정답을 알려주지 않음, 

# 예제 데이터
#새로운 고객 데이터가 있습니다. 이 사람은 차량을 얼마정도 구매 가능한지 예측하시오.
#여자이고, 나이는 38, 연봉은 90000, 카드빚은 2000, 순자산은 500000 일때, 어느정도의 차량을 구매할 수 있을지 예측하시오.
def run_ml_app():

    st.subheader('Machine Learning')
    dir_filename ='data/finalized_model.sav'
    loaded_model = pickle.load(open(dir_filename, 'rb'))
    print(loaded_model)
    
    # Feature Scaling 하기
    # 기존 객체를 사용하기 때문에 transform()으로 함
    # 새로운 데이터를 mm_X 객체를 이용해서 피쳐스케일링
    # new_data = mm_X.transform(new_data)

    df = df_load()
    new_data = np.array([3, 88, 58, 11, 54, 24, 0.26, 22])
    new_data = new_data.reshape(1, -1)
    print(new_data)

    result = loaded_model.predict(new_data)
    print(result)
    

    # st.subheader('...')
    
    # age = st.slider('나이', 10, 100)

    # salary = st.number_input('연봉을 입력하세요', min_value=0)
    # debt = st.number_input('카드빚을 입력하세요', min_value=0)
    # worth = st.number_input('순 자산을 입력하세요', min_value=0)

    if st.button('예측하기'):
        pass