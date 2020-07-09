from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class test():

    def __init__(self):
        self.text_data = ['나는 배가 고프다', '내일 점심 뭐먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']
    def CVect(self):
       count_vectorizer = CountVectorizer()
       count_vectorizer.fit(self.text_data)
       print(count_vectorizer.vocabulary_)
       sentence = [self.text_data[0]] # ['나는 배가 고프다']
       print(count_vectorizer.transform(sentence).toarray())

test().CVect()
