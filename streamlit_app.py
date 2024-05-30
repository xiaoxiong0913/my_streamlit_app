import streamlit as st
import requests
import joblib
import pandas as pd

# 加载模型和Scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('treebag_model.pkl')

# 确认特征名称和顺序
expected_features = list(scaler.feature_names_in_)

# 创建Web应用的标题
st.title('Machine learning-based model predicts 1-year mortality in patients with type A aortic dissection')

# 添加介绍部分
st.markdown("""
## Introduction
This web-based calculator was developed based on the Treebag model with an AUC of 0.94 (95% CI: 0.896-0.966) and a Brier score of 0.128. Users can obtain the 1-year risk of death for a given case by simply selecting the parameters and clicking on the "Predict" button.
""")

# 创建输入表单
st.markdown("## Selection Panel")
st.markdown("Picking up parameters")

with st.form("prediction_form"):
    age = st.slider('Age', min_value=18, max_value=100, value=50)
    wbc = st.slider('WBC (10^9/L)', min_value=2.0, max_value=60.0, value=10.0)
    lym = st.slider('Lym (10^9/L)', min_value=0.05, max_value=7.0, value=1.0)
    co2_bp = st.slider('CO2-Bp(mmol/L)', min_value=3.35, max_value=36.6, value=24.0)
    eos = st.slider('Eos', min_value=0.0, max_value=0.8, value=0.01)
    sbp = st.slider('SBP (mmHg)', min_value=50, max_value=250, value=120)
    beta_blocker = st.selectbox('β-receptor Blocker', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    surgery = st.selectbox('Surgery Therapy', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    # 提交按钮
    submit_button = st.form_submit_button("Predict")

# 定义正常值范围
normal_ranges = {
    "WBC (10^9/L)": (4.5, 10.5),
    "Lym (10^9/L)": (1.0, 3.5),
    "CO2-Bp(mmol/L)": (23.0, 30.0),
    "Eos": (0.02, 0.5),
    "SBP(mmHg)": (110, 130)
}

# 当用户提交表单时
if submit_button:
    # 构建请求数据
    data = {
        "age": age,
        "WBC (10^9/L)": wbc,
        "Lym (10^9/L)": lym,
        "CO2-Bp(mmol/L)": co2_bp,
        "Eos": eos,
        "SBP(mmHg)": sbp,
        "β-receptor blocker(1yes，0no)": beta_blocker,
        "surgery therapy(1yes,0no)": surgery
    }

    # 将数据转换为DataFrame并提供列名，确保特征顺序与拟合时一致
    input_df = pd.DataFrame([data], columns=expected_features)

    # 预处理数据并进行预测
    scaled_features = scaler.transform(input_df)
    prediction = model.predict_proba(scaled_features)[:, 1]  # 获取类别为1的预测概率

    # 显示预测结果，格式化为百分比
    risk_score = prediction[0] * 100
    st.write(f'Prediction : {risk_score:.2f}%')

    # 提供个性化建议
    if risk_score >= 37.9:
        st.markdown(
            "<span style='color:red'>High risk: This patient is classified as a high-risk patient.</span>",
            unsafe_allow_html=True)
        st.write("Personalized Recommendations:")
        # 提供每个特征的调整建议
        for feature, (normal_min, normal_max) in normal_ranges.items():
            value = data[feature]
            if value < normal_min:
                st.markdown(
                    f"<span style='color:red'>{feature}: Your value is {value}. It is lower than the normal range ({normal_min} - {normal_max}). Consider increasing it towards {normal_min}.</span>",
                    unsafe_allow_html=True)
            elif value > normal_max:
                st.markdown(
                    f"<span style='color:red'>{feature}: Your value is {value}. It is higher than the normal range ({normal_min} - {normal_max}). Consider decreasing it towards {normal_max}。</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"{feature}: Your value is within the normal range ({normal_min} - {normal_max}).")

        # 药物治疗建议
        if beta_blocker == 0:
            st.write("Consider using β-receptor blocker medication.")
        if surgery == 0:
            st.write("Consider undergoing surgery therapy.")
    else:
        st.markdown(
            "<span style='color:green'>Low risk: This patient is classified as a low-risk patient.</span>",
            unsafe_allow_html=True)
