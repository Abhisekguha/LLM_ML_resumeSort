import os
import pickle
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Combobox
from PIL import ImageTk, Image
import pandas as pd
import re
import string
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Rest of your code...


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained classifiers and vectorizers
with open("pickles/le.pickle", "rb") as le_file:
    le = pickle.load(le_file)

with open("pickles/word_vectorizer.pickle", "rb") as word_vectorizer_file:
    word_vectorizer = pickle.load(word_vectorizer_file)

with open("pickles/clf.pickle", "rb") as clf_file:
    clf = pickle.load(clf_file)

# Function to clean resume text


def clean_resume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# Function to predict resume category


def predict_category(resume_text):
    cleaned_resume = clean_resume(resume_text)
    prediction = clf.predict_proba(
        word_vectorizer.transform([cleaned_resume]))[0]
    predicted_category = le.inverse_transform([prediction.argmax()])[0]
    return predicted_category

# Function to process resumes in batches


def process_resumes_in_batches(resumes, job_category):

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            process_single_resume, resume, job_category) for resume in resumes]
        results = [future.result() for future in futures]
        return results

# Function to process a single resume and predict category


def process_single_resume(resume, job_category):
    predicted_category = predict_category(resume)
    return (predicted_category, resume)


# Function to open and read an application file


def open_application_file():
    file_path = filedialog.askopenfilename(
        title="Open Application File", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        chunksize = 10 ** 6  # process 1 million rows at a time
        for chunk in pd.read_csv(file_path, chunksize=chunksize, encoding='latin1'):
            # Check if 'Resume' column exists in the DataFrame
            if 'Resume' not in chunk.columns:
                messagebox.showerror(
                    "Error", "The 'Resume' column does not exist in the application file.")
                return

            cv_list.delete(1.0, tk.END)  # Clear previous results

            job_category = combo_job_category.get()
            filtered_cv_count = 0

            # Batch processing resumes
            batch_resumes = chunk['Resume'].tolist()
            batch_results = process_resumes_in_batches(
                batch_resumes, job_category)

            for predicted_category, candidate_resume in batch_results:
                if predicted_category == job_category:
                    cv_list.insert(tk.END, f"Candidate {i+1}\n")
                    cv_list.insert(
                        tk.END, f"Predicted Category: {predicted_category}\n")
                    cv_list.insert(tk.END, "=======================\n")
                    cv_list.insert(tk.END, candidate_resume + '\n\n')
                    filtered_cv_count += 1

            if filtered_cv_count == 0:
                cv_list.insert(
                    tk.END, "No CVs found for the selected job category.")


# Function to open and read an application file


def open_application_file():
    file_path = filedialog.askopenfilename(
        title="Open Application File", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        application_df = pd.read_csv(file_path, encoding='latin1')

        # Check if 'Resume' column exists in the DataFrame
        if 'Resume' not in application_df.columns:
            messagebox.showerror(
                "Error", "The 'Resume' column does not exist in the application file.")
            return

        cv_list.delete(1.0, tk.END)  # Clear previous results

        job_category = combo_job_category.get()
        filtered_cv_count = 0

        # Batch processing resumes
        batch_size = 20
        num_candidates = len(application_df)
        for i in range(0, num_candidates, batch_size):
            batch_resumes = application_df['Resume'][i:i+batch_size].tolist()
            batch_results = process_resumes_in_batches(
                batch_resumes, job_category)

            for predicted_category, candidate_resume in batch_results:
                if predicted_category == job_category:
                    cv_list.insert(tk.END, f"Candidate {i+1}\n")
                    cv_list.insert(
                        tk.END, f"Predicted Category: {predicted_category}\n")
                    cv_list.insert(tk.END, "=======================\n")
                    cv_list.insert(tk.END, candidate_resume + '\n\n')
                    filtered_cv_count += 1

        if filtered_cv_count == 0:
            cv_list.insert(
                tk.END, "No CVs found for the selected job category.")


warnings.filterwarnings('ignore')

resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()
resumeDataSet['Category'].value_counts()

plt.figure(figsize=(5, 5))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=resumeDataSet)


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(
    lambda x: cleanResume(x))
nltk.download('stopwords')
nltk.download('punkt')


oneSetOfStopWords = set(stopwords.words('english')+['``', "''"])
totalWords = []
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for i in range(0, 160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
    from sklearn.model_selection import train_test_split

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print("Feature completed .....")

X_train, X_test, y_train, y_test = train_test_split(
    WordFeatures, requiredTarget, random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(
    clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(
    clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" %
      (clf, metrics.classification_report(y_test, prediction)))

predicted = clf.predict_proba(word_vectorizer.transform(["""
•	To expose myself to new areas, learning and to improve my skills and to collaborate purposefully across boundaries.
•	To join the Banking world to provide better solutions and aspiring CIFC Candidate and to join the customer service role to make the customer’s life at ease.


Current Organization: RBS Services India private Limited (NatWest Group)
Designation: Senior Customer Service & Operations Analyst
Experience: 2 Years 9 Months
Job role: To board the customer into the product/services that they availed and to attend the concerns and queries from various stake holders by following service line agreement. Maintains and records customer’s banking account information in an IT tool.


Department	Roles & Responsibilities
Mortgage Operations	Funds have to be investigated and posting of transactions on the data base systems. KYC and all other documents have to be indexed with the customer account and also to check the affordability.

Financial Crime Operations (Anti- Money Laundering)	Analyst has to look for opportunity to simplify the bank’s process, responding to queries from stakeholders promptly and within our service targets, perform in well fast paced environment and should have presentational skills to deliver to deadline, authorize and investigate transactions, identify red flags, unusual & suspicious pattern in the wire transfer and generate full report on high risk transaction with evidences. Its main objective is to combat money laundering activities, financial crimes occurred using financial institution to avoid reputational damage and fines from regulatory body.



•	Agile Level 1 certified
•	Diploma in RPA (Robotic Process Automation – Ui Path Training)
•	Best volunteer Award for community service
•	Living Our Values (LOV) Award for serving Customers
•	Simplify the Bank (STB) Ideas 7 logged and 1 implemented
•	Spot Ovation for serving Customer & Thinking long Term – 2 times
•	Bronze Ovation for excellent Team work



Degree/Course	Institution/ School	Board/ University	Year of Passing	Percentage
PGDM in Big Data Analytics	Lambton College, Mississauga	Ontario Public College	Currently Pursuing	-
B.Sc. Mathematics  	Loyola College – Autonomous	Madras University	2019	89.80%



•	MS Excel
•	MS word
•	MS PowerPoint



•	Good understanding of risks involved in a role and has the good ability to mitigate it.
•	Good knowledge in banking and also have end to end process knowledge about mortgage journey
•	Good knowledge in customer centric roles and conflicts of interest would be handled in the best interest of the customer.
•	Exhibited leadership quality in Lean tool project.




Days	Monday	Wednesday	Thursday	Friday	Saturday	Sunday
Availability	 From anytime
     Till
5:30 PM	Anytime	Anytime	    From anytime
     Till
5:30 PM	Anytime
	Anytime

"""]))
# le.inverse_transform([4])[0]
predicted[0]

# new block
# Create the 'pickles' directory if it doesn't exist
if not os.path.exists('pickles'):
    os.makedirs('pickles')

# Save the label encoder
save_label_encoder = open("pickles/le.pickle", "wb")
pickle.dump(le, save_label_encoder)
save_label_encoder.close()

# Save the word vectorizer
save_word_vectorizer = open("pickles/word_vectorizer.pickle", "wb")
pickle.dump(word_vectorizer, save_word_vectorizer)
save_word_vectorizer.close()

# Save the classifier
save_classifier = open("pickles/clf.pickle", "wb")
pickle.dump(clf, save_classifier)
save_classifier.close()


# GUI Components
app = tk.Tk()
app.title("Candidate CV Sorter")

lbl_instructions = tk.Label(app, text="Select the Job Category:")
lbl_instructions.pack()

combo_job_category = Combobox(
    app, values=["Data Science",
                 "HR",
                 "Advocate",
                 "Arts",
                 "Web Designing",
                 "Mechanical Engineer",
                 "Sales",
                 "Health and fitness",
                 "Civil Engineer",
                 "Java Developer",
                 "Business Analyst",
                 "SAP Developer",
                 "Automation Testing",
                 "Electrical Engineering",
                 "Operations Manager",
                 "Python Developer",
                 "DevOps Engineer",
                 "Network Security Engineer",
                 "PMO",
                 "Database",
                 "Hadoop",
                 "ETL Developer",
                 "DotNet Developer",
                 "Blockchain",
                 "Testing"
                 ])
combo_job_category.pack()

btn_open_application = tk.Button(
    app, text="Open Application File", command=open_application_file)
btn_open_application.pack()

cv_list = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=80, height=30)
cv_list.pack()

app.mainloop()
