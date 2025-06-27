import streamlit as st
import pandas as pd
import joblib
import numpy as np # تأكد من وجود numpy
# تأكد من نوع موديلك (RandomForestClassifier أو غيره)
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="توقع تذبذب عملاء تأجير السيارات",
    page_icon="🚗",
    layout="centered"
)

# --- Load The Model ---
# هذا الموديل تم تدريبه على بيانات عملاء الاتصالات، وليس تأجير السيارات.
# التنبؤات ستكون غير دقيقة لهذا السبب.
try:
    model = joblib.load('models/churn_classifier_model.joblib')
except FileNotFoundError:
    st.error("Model file not found! Please make sure 'churn_classifier_model.joblib' is in the 'models' folder.")
    st.stop()

# --- Define Expected Columns from Training Data ---
# هذه الأعمدة هي من موديل تدرب على بيانات عملاء الاتصالات.
# ستحتاج لتحديث هذه القائمة بالكامل عند تدريب موديل جديد على بيانات تأجير السيارات.
expected_columns = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_Yes',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_No internet service',
    'OnlineSecurity_Yes',
    'OnlineBackup_No internet service',
    'OnlineBackup_Yes',
    'DeviceProtection_No internet service',
    'DeviceProtection_Yes',
    'TechSupport_No internet service',
    'TechSupport_Yes',
    'StreamingTV_No internet service',
    'StreamingTV_Yes',
    'StreamingMovies_No internet service',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# --- App Title and Description ---
st.title('🚗 توقع تذبذب عملاء تأجير السيارات')
st.write(
    "يساعد هذا التطبيق شركة تأجير السيارات على التنبؤ بالعملاء المعرضين لخطر التوقف عن التأجير بناءً على سلوكهم وتفاصيل تعاملهم."
)
st.write("---")

# --- User Inputs (مدخلات جديدة لشركة تأجير السيارات) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("معلومات العميل")
    customer_tenure = st.number_input('مدة التعامل مع الشركة (بالأشهر)', min_value=0, max_value=200, value=12, step=1)
    avg_rental_duration = st.number_input('متوسط مدة التأجير (بالأيام)', min_value=1, max_value=30, value=3, step=1)
    num_rentals = st.number_input('عدد مرات التأجير الإجمالي', min_value=0, max_value=100, value=5, step=1)

with col2:
    st.subheader("تفاصيل التأجير")
    favorite_car_type = st.selectbox('نوع السيارة المفضل', ['اقتصادي', 'سيدان', 'دفع رباعي', 'فان', 'فاخرة', 'رياضية', 'لم يحدد'])
    rental_location = st.selectbox('موقع التأجير الرئيسي', ['المطار', 'وسط المدينة', 'فندق', 'محلي', 'آخر'])
    loyalty_program = st.radio('عضو في برنامج الولاء؟', ['نعم', 'لا'])

with col3:
    st.subheader("معلومات أخرى")
    avg_monthly_spend = st.slider('متوسط الإنفاق الشهري ($)', 50.0, 1000.0, 250.0)
    last_rental_days_ago = st.number_input('آخر تأجير (منذ كم يوم)', min_value=0, max_value=365, value=30, step=1)
    # يمكن استبدال "الشكاوى السابقة" بمعلومات أكثر تفصيلاً إذا كانت متوفرة
    complaints_filed = st.radio('هل لديه شكاوى سابقة؟', ['نعم', 'لا'])


# --- Prediction Logic ---
if st.button('توقع التذبذب', type="primary"):
    # بناء قاموس من المدخلات الخام الجديدة
    # ملاحظة: هذه المدخلات لا تتطابق مع الموديل الحالي.
    # هذا الجزء يحتاج لتعديل ليتناسب مع الأعمدة بعد get_dummies للموديل الجديد.
    input_data = {
        'customer_tenure': customer_tenure,
        'avg_rental_duration': avg_rental_duration,
        'num_rentals': num_rentals,
        'favorite_car_type': favorite_car_type,
        'rental_location': rental_location,
        'loyalty_program': loyalty_program,
        'avg_monthly_spend': avg_monthly_spend,
        'last_rental_days_ago': last_rental_days_ago,
        'complaints_filed': complaints_filed
    }

    # تحويل المدخلات إلى DataFrame
    input_df = pd.DataFrame([input_data])

    # تطبيق get_dummies على بيانات الإدخال الجديدة
    # يجب أن تكون أسماء الأعمدة هنا هي الأسماء الأصلية للميزات الفئوية الجديدة.
    input_df_processed = pd.get_dummies(input_df, columns=[
        'favorite_car_type', 'rental_location', 'loyalty_program', 'complaints_filed'
    ], drop_first=True) # drop_first=True هو الأكثر شيوعاً

    # ضمان أن جميع الأعمدة المتوقعة موجودة، وتعيين القيم المفقودة إلى 0
    # هذا حاسم جداً: يجب أن تكون expected_columns هنا هي أعمدة موديل الاتصالات الحالي
    # لذا، هذا الجزء سيجعل المدخلات الجديدة تتماشى مع أعمدة موديل الاتصالات، مما سيؤدي
    # إلى تنبؤات غير صحيحة لأن الموديل ليس مدرباً على هذه البيانات.
    final_input_df = pd.DataFrame(columns=expected_columns) #expected_columns من موديل الاتصالات
    final_input_df = pd.concat([final_input_df, input_df_processed], ignore_index=True)
    final_input_df = final_input_df.fillna(0) # املأ القيم المفقودة بـ 0
    final_input_df = final_input_df.astype(float) # تأكد أن الأعمدة كلها أرقام

    # إعادة ترتيب الأعمدة لتطابق ترتيب أعمدة موديل الاتصالات بالضبط
    final_input_df = final_input_df[expected_columns]

    # إجراء التنبؤ باستخدام النموذج المدرب (موديل الاتصالات الحالي)
    churn_prediction = model.predict(final_input_df)
    churn_proba = model.predict_proba(final_input_df)[:, 1] # احتمالية التذبذب

    st.write("---")
    st.subheader("نتيجة التنبؤ:")

    # رسالة تحذير واضحة جداً للمستخدم
    st.warning("⚠️ ملاحظة هامة: هذا التنبؤ غير دقيق! الموديل الحالي تم تدريبه على بيانات عملاء الاتصالات وليس تأجير السيارات. التنبؤات ستكون عشوائية.", icon="⚠️")

    if churn_prediction[0] == 1:
        st.error(f"⚠️ احتمال كبير لتذبذب العميل! (الاحتمالية: {churn_proba[0]:.2%})")
        st.write("بناءً على موديل الاتصالات. للحصول على تنبؤات دقيقة، يجب تدريب موديل جديد على بيانات تأجير السيارات.")
    else:
        st.success(f"✅ احتمال منخفض لتذبذب العميل. (الاحتمالية: {churn_proba[0]:.2%})")
        st.write("بناءً على موديل الاتصالات. للحصول على تنبؤات دقيقة، يجب تدريب موديل جديد على بيانات تأجير السيارات.")

    st.write("---")
    st.info("الخطوة التالية: تدريب موديل ذكاء اصطناعي جديد على بيانات عملاء تأجير السيارات الحقيقية، وتحديث قائمة الأعمدة المتوقعة (expected_columns) في الكود.")


