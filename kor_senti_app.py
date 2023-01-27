pip install transformers
import io
import streamlit as st
import numpy as np
from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      "snunlp/KR-FinBert-SC")

st.header('Оценка эмоциональной окраски текстов на корейском языке')
form = st.form(key='my_form')
text = form.text_input(label='Введите текст')
submit_button = form.form_submit_button(label='Выполнить')

def create_score_text(clsfr, txt):
    result = clsfr(txt)
    st.write('**Оценка фразы:** ' + result[0].get('label', 'no res'))
    if result[0]['score'] >= 0.75:
        return '**Степень достоверности:** достоверный результат (score >= 0.75)'
    else:
        return '**Степень достоверности:** точность невысокая (score < 0.75)'

if submit_button:
    res_func = create_score_text(classifier,text)
    st.write(res_func)