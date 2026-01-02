import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

okt=Okt() #형태소분석기

def korean_tokenizer(text):#어간추출함수
    tokens=okt.pos(text,stem=True)#stem=True로 좋은->좋다 먹고->먹다로 변환
    return[word for word,pos in tokens if pos in['Noun']]#명사,동사,형용사만 필터링해서 리스트변환

tfidf=TfidfVectorizer(tokenizer=korean_tokenizer)#모델에 함수넣어주기

train=pd.read_csv("C:\\Users\\jackj\\Desktop\\개인프로젝트\\backend\\restaurants.csv")
tfidf_matrix=tfidf.fit_transform(train['review'])

#사용자입력받기
print("어떤분위기의맛집을찾니")
user_input=input()

user_vector=tfidf.transform([user_input])

print(user_vector)

similarity = cosine_similarity(user_vector, tfidf_matrix)
train['score'] = similarity[0]
result = train.sort_values(by='score', ascending=False).head(5)
print(result[['name', 'review', 'score']])