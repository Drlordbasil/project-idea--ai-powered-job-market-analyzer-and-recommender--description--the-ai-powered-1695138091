import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import random

class JobMarketScraper:
    def scrape_job_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        job_elements = soup.find_all('div', class_='job')

        job_data = []
        
        for job_element in job_elements:
            title = job_element.find('h3').text.strip()
            company = job_element.find('p', class_='company').text.strip()
            location = job_element.find('p', class_='location').text.strip()
            salary = job_element.find('p', class_='salary').text.strip()

            job_data.append({'title': title, 'company': company, 'location': location, 'salary': salary})

        return job_data


class JobMarketAnalyzer:
    def __init__(self):
        self.job_data = []

    def cluster_job_data(self):
        job_df = pd.DataFrame(self.job_data)
        vectorizer = CountVectorizer()
        job_vectors = vectorizer.fit_transform(job_df['title'])

        # Apply KMeans clustering algorithm
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(job_vectors)

        # Assign cluster labels to job dataframe
        job_df['cluster'] = kmeans.labels_
        return job_df

    def train_classification_model(self, data):
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)

        model = MultinomialNB()

        # Prepare training and testing data for classification model
        X_train = train_data.drop(['company'], axis=1)
        y_train = train_data['company']
        X_test = test_data.drop(['company'], axis=1)
        y_test = test_data['company']

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train['title'])
        X_test = vectorizer.transform(X_test['title'])

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted)
        return model, accuracy

    def predict_salary(self, data, title):
        regression_model = LinearRegression()
        X = data.drop(['salary'], axis=1)
        y = data['salary']

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X['title'])

        regression_model.fit(X, y)
        title_vector = vectorizer.transform([title])

        predicted_salary = regression_model.predict(title_vector)
        return predicted_salary[0]

    def analyze_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(text)
        return sentiment_scores['compound']

    def recommend_career(self, skills, experience, aspirations):
        recommended_career = random.choice(['Data Scientist', 'Software Engineer', 'UI/UX Designer'])
        return recommended_career

    def analyze_skill_gap(self, user_skills, job_requirements):
        user_skills_set = set(user_skills)
        job_requirements_set = set(job_requirements)

        missing_skills = job_requirements_set - user_skills_set
        return list(missing_skills)

    def analyze_events(self, industry):
        upcoming_events = ['Conference A', 'Webinar B']
        return upcoming_events

    def find_mentors(self, aspirations):
        mentors = ['Mentor A', 'Mentor B', 'Mentor C']
        return mentors

    def run(self):
        scraper = JobMarketScraper()
        self.job_data = scraper.scrape_job_data('https://example.com/job-market')

        job_df = self.cluster_job_data()
        classification_model, accuracy = self.train_classification_model(job_df)

        user_skills = ['Python', 'Machine Learning']
        user_experience = '2 years'
        user_aspirations = 'Data Science'

        recommended_career = self.recommend_career(user_skills, user_experience, user_aspirations)

        job_requirements = ['Python', 'Machine Learning', 'Data Analysis']
        skill_gap = self.analyze_skill_gap(user_skills, job_requirements)

        industry_events = self.analyze_events('Data Science')

        mentors = self.find_mentors(user_aspirations)

        print(job_df.head())
        print(f"Model accuracy: {accuracy}")
        print(f"Recommended career: {recommended_career}")
        print(f"Skill gap: {skill_gap}")
        print(f"Upcoming events: {industry_events}")
        print(f"Mentors: {mentors}")

job_market_analyzer = JobMarketAnalyzer()
job_market_analyzer.run()