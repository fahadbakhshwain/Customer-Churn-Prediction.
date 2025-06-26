import streamlit as st
import pandas as pd
import joblib
import numpy as np # تأكد من وجود numpy
from sklearn.ensemble import RandomForestClassifier # تأكد من نوع موديلك

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="centered"
)

# --- Load The Model ---
try:
    model = joblib.load('models/churn_classifier_model.joblib')
except FileNotFoundError:
    st.error("Model file not found! Please make sure 'churn_classifier_model.joblib' is in the 'models' folder.")
    st.stop()

# --- Define Expected Columns from Training Data ---
# هذا الجزء حاسم: يجب أن يتضمن جميع الأعمدة بعد pd.get_dummies على بيانات التدريب
# ستحتاج للحصول على قائمة الأعمدة النهائية من الـ Notebook الخاص بك
# بعد تطبيق pd.get_dummies() على البيانات الكاملة.
# مثال (يجب تعديله ليتناسب مع أعمدتك الحقيقية):
expected_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']


    # أضف هنا جميع الأعمدة الأخرى التي تم إنشاؤها بواسطة get_dummies
    # لكل الميزات الفئوية التي استخدمتها في تدريب الموديل.
    # يجب أن تكون بنفس الترتيب الذي دخلت به إلى الموديل.
]

# --- App Title and Description ---
st.title('👋 Customer Churn Predictor')
st.write(
    "This app predicts whether a customer is likely to churn based on their service usage and contract details. "
    "Fill in the customer's information below to get a prediction."
)
st.write("---")

# --- User Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input('Tenure (in months)', min_value=0, max_value=72, value=1, step=1)
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

with col2:
    monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=70.0)
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

with col3:
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    pmt_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])

# --- Prediction Logic ---
if st.button('Predict Churn', type="primary"):
    # بناء قاموس من المدخلات الخام
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract, # تبقى كنص هنا
        'InternetService': internet_service, # تبقى كنص هنا
        'OnlineSecurity': online_security, # تبقى كنص هنا
        'TechSupport': tech_support, # تبقى كنص هنا
        'PaymentMethod': pmt_method, # تبقى كنص هنا
        'PaperlessBilling': paperless_billing # تبقى كنص هنا
        # أضف هنا أي ميزات أخرى استخدمتها في النموذج
        # تأكد من أن أسماء المفاتيح هنا تتطابق مع أسماء الأعمدة الأصلية في بياناتك
    }

    # تحويل المدخلات إلى DataFrame
    input_df = pd.DataFrame([input_data])

    # تطبيق get_dummies على بيانات الإدخال
    # هذا يضمن أن يتم إنشاء نفس الأعمدة التي كانت في بيانات التدريب
    input_df_processed = pd.get_dummies(input_df, columns=[
        'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'PaperlessBilling'
        # أضف هنا جميع أسماء الأعمدة الفئوية الأصلية التي قمت بعمل get_dummies لها في التدريب
    ], drop_first=True)

    # ضمان أن جميع الأعمدة المتوقعة موجودة، وتعيين القيم المفقودة إلى 0
    # هذا حاسم جداً لـ get_dummies، حيث قد لا تظهر كل الفئات في إدخال واحد
    # مثال: إذا لم يختار المستخدم 'Fiber optic'، فإن 'InternetService_Fiber optic' لن تنشأ
    # يجب أن نضمن أنها موجودة في الـ DataFrame بـ 0
    final_input_df = pd.DataFrame(columns=expected_columns)
    final_input_df = pd.concat([final_input_df, input_df_processed], ignore_index=True)
    final_input_df = final_input_df.fillna(0) # املأ القيم المفقودة بـ 0
    final_input_df = final_input_df.astype(float) # تأكد أن الأعمدة كلها أرقام

    # إعادة ترتيب الأعمدة لتطابق ترتيب أعمدة التدريب بالضبط
    # (هذه خطوة حاسمة جداً لعمل النموذج بشكل صحيح)
    # تأكد أن 'expected_columns' (المذكورة في بداية الكود) هي بالترتيب الصحيح.
    final_input_df = final_input_df[expected_columns]


    # إجراء التنبؤ باستخدام النموذج المدرب
    churn_prediction = model.predict(final_input_df)
    churn_proba = model.predict_proba(final_input_df)[:, 1] # احتمالية التذبذب

    st.write("---")
    st.subheader("نتيجة التنبؤ:")

    if churn_prediction[0] == 1:
        st.error(f"⚠️ احتمال كبير لتذبذب العميل! (الاحتمالية: {churn_proba[0]:.2%})")
        st.write("يُنصح بالتدخل السريع للاحتفاظ بهذا العميل.")
    else:
        st.success(f"✅ احتمال منخفض لتذبذب العميل. (الاحتمالية: {churn_proba[0]:.2%})")
        st.write("يُتوقع أن يبقى العميل.")

    st.write("---")
    st.info("ملاحظة: دقة التنبؤ تعتمد على جودة بيانات التدريب ومدى تطابق أعمدة المدخلات مع أعمدة التدريب الأصلية.")