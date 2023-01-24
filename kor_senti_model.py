from transformers import pipeline

classifier = pipeline("sentiment-analysis",   
                      "snunlp/KR-FinBert-SC")

print(classifier("C쇼크에 멈춘 흑자비행…대한항공 1분기 영업적자 566억"))