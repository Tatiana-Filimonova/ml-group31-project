from transformers import pipeline
import streamlit as st

classifier = pipeline("sentiment-analysis",   
                      "snunlp/KR-FinBert-SC")

st.write(classifier(st.text_input('Введите фразу для оценки...')))