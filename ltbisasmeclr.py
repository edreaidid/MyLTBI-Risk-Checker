# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:37:13 2023

@author: Edre MA
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
import shap

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

import hydralit_components as hc
from streamlit_extras.switch_page_button import switch_page

# define what option labels and icons to display
option_data = [
   {'icon': "fas fa-tachometer-alt", 'label':"logout"},
   {'icon': "bi bi-hand-thumbs-up", 'label':"Home"},
   {'icon':"fa fa-question-circle",'label':"Our team"},
   {'icon': "far fa-chart-bar", 'label':"LTBI info"},
   {'icon': "fa fa-unlock-alt", 'label':"Risk calculator"}
]

# override the theme, else it will use the Streamlit applied theme
over_theme = {'txc_inactive': 'white','menu_background':'#999966','txc_active':'yellow','option_active':'blue'}
font_fmt = {'font-class':'h2','font-size':'150%'}

# display a horizontal version of the option bar
op = hc.option_bar(option_definition=option_data,key='PrimaryOption',override_theme=over_theme,font_styling=font_fmt,horizontal_orientation=True,first_select=4)



st.write("""
# Malaysian LTBI Risk Checker for HCW

The logistic regression model uses real hospital data to determine likelihood of LTBI
""")

with st.expander("User input parameters"):
    def user_input_features():
        age = st.slider('AGE', 18, 85, 30)
        indexoptions = ["1","2","3"]
        index=st.selectbox("INDEX: HCW(1),screening(2),patient(3)",options=indexoptions)
        sexoptions = ["1","2"]
        sex=st.selectbox('SEX',options=sexoptions)
        postoptions=["1","2","3"]
        post = st.selectbox('POST',options=postoptions)
        deptoptions=["1","2"]
        dept = st.selectbox('DEPT',options=deptoptions)
        testoptions=["1","2"]
        test = st.selectbox('TEST',options=testoptions)
        data = {'INDEX: HCW(1),screening(2),patient(3)': index,
                'SEX': sex,
                'AGE': age,
                'POST': post,
                'DEPT': dept,
                'TEST': test}
        features = pd.DataFrame(data, index=[0])
        return features
    z = user_input_features()
    st.write("Footnote:")


df = pd.read_csv('ltbi_ML_V5.csv')

X = df[["INDEX","SEX","AGE","POST","DEPT","TEST"]]
Y= df["LTBI"]

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=35)
clf = LogisticRegression()
model=clf.fit(X, Y)

y_pred = model.predict(x_test)
modelacc=accuracy_score(y_test, y_pred)


#Now we set the web app to predict  using our trained model

prediction = model.predict(z)
dfpred = pd.DataFrame(prediction)
prediction_proba = clf.predict_proba(z)


ltbi = dfpred.replace([0, 1], ["N", "Y"])

    
#writing prediction result
st.subheader('Prediction')
st.info(f"Your predicted LTBI status : {ltbi}")
st.write(f"model accuracy: {(round(modelacc,2))*100} %")
   
#writing prediction probability, the higher prediction will be chosen as above
st.subheader('Prediction Probability (%)')
predproba=prediction_proba*100 
dfpredproba = pd.DataFrame(predproba, columns=['N', 'Y']) 
styler2 = dfpredproba.style.hide()
st.write(styler2.to_html(), unsafe_allow_html=True)

#interpretation using explainable AI (XAI)
st.write("")

shapgraphic=st.button("Generate feature importance chart")
if shapgraphic:
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    xai=shap.summary_plot(shap_values, X)
    st.pyplot(xai)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write('Footnote : The right x axis signifies higher LTBI risk, left lower risk. Y axis is the factors')

st.text('©️ EA2023')
