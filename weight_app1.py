# 남자와 여자 모두 출력 

import streamlit as st
import numpy as np
import joblib

st.title("신체 정보를 이용한 몸무게 예측 머신러닝 모델")
st.write("신체 정보를 입력하면 몸무게를 예측합니다.")

model_male = joblib.load("weight_model_male.pkl")
model_female = joblib.load("weight_model_female.pkl")

gender = st.radio("성별 선택", ["남자", "여자"])

st.sidebar.header("머신러닝 모델 설계 실습 (다중회귀)")

if gender == "남자":
    height = st.slider("키 (cm)", 140.0, 190.0, 170.0)
    waist = st.slider("허리 둘레 (cm)", 50.0, 120.0, 80.0)
    waist2 = st.slider("엉덩이 둘레 (mm)", 85, 120, 100)

    X = np.array([[height, waist, waist2]])
    model = model_male

else:
    neck = st.slider("목 둘레 (cm)", 20.0, 50.0, 30.0)
    waist = st.slider("허리 둘레 (cm)", 40.0, 120.0, 80.0)    
    hip = st.slider("엉덩이 둘레 (cm)", 80.0, 120.0, 90.0)

    X = np.array([[neck, waist, hip]])
    model = model_female

if st.button("몸무게 예측하기"):
    prediction = model.predict(X)
    st.success(f"예측 몸무게 : {prediction[0]:.1f} kg")