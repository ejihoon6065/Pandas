from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#text_data = ['나는 배가 고프다', '내일 점심 뭐먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']
def datatest():
    text_data = ['나는 배가 고프다', '오늘 점심 뭐먹지', '오늘 공부 해야겠다', '점심 먹고 공부 해야지']
    return  text_data

def tfidfVect():
    a = datatest()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(a)
    print(tfidf_vectorizer.vocabulary_)
    sentence = [a[3]]  # ['점심 먹고 공부 해야지']
    print(tfidf_vectorizer.transform(sentence).toarray())

tfidfVect()
