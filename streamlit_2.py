import streamlit as st
import pandas as pd
# import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import io
# from docx import Document
import nltk
import re
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid
# import openai
import concurrent.futures
# import joblib
import plotly.express as px
from pdfminer.high_level import extract_text
# from transformers import pipeline
from openai import OpenAI
import os
# import pyttsx3
# from streamlit_TTS import auto_play, text_to_audio
# from tts import TextToSpeech
# from transformers import pipeline
# import scipy

# txt2speech = TextToSpeech()

# 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
# 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

client = OpenAI(api_key=st.secrets.secrets['API_KEY'])

nltk.download('stopwords')
nltk.download('punkt')



# speaker = pyttsx3.init()
titles = []
count_matrix = []
cos_similarity = []
indexes = []

stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words='english')
model_name = "deepset/roberta-base-squad2"
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)



# synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")


def extract_job_titles_1(text):
    # Define a regular expression pattern to match potential job titles
    pattern = r'\b(?:UI/UX\s*(?:Designer|Developer)|Marketing\s*Specialist|Programmer|Call\s*Center\s*Agent|Back[-\s]*End\s*Developer|UI\s*designer|Junior\s*Android\s*Developer|MOBILE\s*APPLICATION\s*DEVELOPER|UX\s*designer|Graphic\s*designer|Front[-\s]*End\s*Developer|Software\s*Developer|Technical\s*Writer|Mobile\s*Developer|Flutter\s*Developer|DevOps|Sales\s*Manager|Machine\s*Learning\s*(?:Developer|Engineer|ML\s*Engineer|Data\s*Scientist|ML\s*Developer)|Python\s*Developer|Data\s*Analyst|Go\s*Developer|Golang\s*Developer|Full[-\s]*Stack\s*(?:Developer)?|System\s*Analyst|Cyber[-\s]*Security\s*Engineer|React\s*js\s*Developer|Quality\s*Assurance\s*(?:Developer)?|Software\s*Solution\s*Architect|Digital\s*Marketing\s*Expert|Listing\s*Manager|Real\s*Estate\s*[-\s]*Sales\s*and\s*Leasing\s*Agent|Video\s*Editor|Photographer|Real\s*estate\s*Listing\s*Coordinator|Team\s*Leader\s*[-\s]*Real\s*Estate|Leasing\s*Agent|Sales\s*and\s*Leasing\s*Agent|Sales\s*Agent)\b'

    # Use the findall function to extract all occurrences of job titles in the text
    job_titles = re.findall(pattern, text, flags=re.IGNORECASE)

    return job_titles



@st.cache
def get_main_phones(raw_phone_numbers):
    
    phone_pattern = re.compile(r'\d*\.?\d+')
    phone_numbers = [match.group() for match in phone_pattern.finditer(raw_phone_numbers)]
    for phone in phone_numbers:
        if len(phone) > 8:
            return phone


def show(df):
    try:
        gd = GridOptionsBuilder.from_dataframe(df.iloc[:, 1:])
        gd.configure_selection(selection_mode='single'
                            , use_checkbox=True)
        gd.configure_grid_options(alwaysShowVerticalScroll = True, enableRangeSelection=True, pagination=True)
        grid_options = gd.build()

        grid_table = AgGrid(df
                            , gridOptions=grid_options
                            , height=600)
        
        values = list(grid_table['selected_rows'][0].values())[1:]
        keys = list(grid_table['selected_rows'][0].keys())[1:]
    
        record = {}
        for key, value in zip(keys, values):
            record[key] = value

        return record
    except:
        pass

    
@st.cache
def filter_text(text_content):
    tokens = word_tokenize(text_content)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    text = ' '.join(filtered_tokens)
    translation_table = str.maketrans('', '', string.punctuation)
    text = text.translate(translation_table)

    return text.lower()


# @st.cache
# def read_docx(file):
#     doc = Document(file)
#     text_content = ""
#     for paragraph in doc.paragraphs:
#         text_content += ' '.join(paragraph.text.split('\n'))

#     new_text = ' '.join(text_content.split('\n'))
#     return new_text.lower().strip()


@st.cache
def read_pdf(file):
    text = extract_text(pdf_file=file)
    new_text = ' '.join(text.split('\n'))
    return new_text.strip().lower()



@st.cache
def get_files_content(cv):
    file_bytes = b''

    if cv.name.split('.')[-1] == 'pdf':
        file_bytes = io.BytesIO(cv.read())
        cleaned_text = read_pdf(file_bytes)

    # elif cv.name.split('.')[-1] == 'docx' or cv.split('.')[-1] == 'doc':
    #     file_bytes = io.BytesIO(cv.read())
        # cleaned_text = read_docx(file_bytes)

    postition = ''
    try:
        postition = extract_job_titles_1(cleaned_text)[0]
    except:
        pass

    email_pattern = r'[\w\.-]+@[\w\.-]+'

    emails = ''
    try:
        emails = re.findall(email_pattern, cleaned_text)[0]
    except:
        pass

    phone_number = get_main_phones(''.join(cleaned_text.split()).lower())
    return postition, phone_number, emails, file_bytes, cleaned_text


@st.cache(allow_output_mutation=True)
def go_to_threads(files):
    df = pd.DataFrame()
    files_bytes = []
    position_list = []
    phone_number_list = []
    email_list = []


    for cv in files:
        print(cv)
        print(cv.name)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread = executor.submit(get_files_content, (cv))
            result_value = thread.result()
            
            position_list.append(result_value[0])
            phone_number_list.append(result_value[1])
            email_list.append(result_value[2])
            files_bytes.append(result_value[3])
            cleaned_text =  result_value[4]
            titles.append(cv.name)
            df = pd.concat([df, pd.DataFrame([cleaned_text.lower()], columns=['cv'])], ignore_index=True)
            
            # print('--------------------------')

    df['title'] = titles
    df['bytes'] = files_bytes
    df['postition'] = position_list
    df['phone_number'] = phone_number_list
    df['email'] = email_list

    return df


# @st.cache
def get_scores(cv, clean_keywords):
    matrix = vectorizer.fit_transform([cv, clean_keywords])
    scores = cosine_similarity(matrix)[0][1]

    return scores

@st.cache
def read_text(text):
    # music = synthesiser(text, forward_params={"do_sample": True})

    # scipy.io.wavfile.write("./audio.wav", rate=music["sampling_rate"], data=music["audio"])

    speaker.save_to_file(text, 'audio.wav')
    speaker.runAndWait()


st.title('CV Filteration')

files = st.file_uploader('Upload files', accept_multiple_files=True, type=['PDF', 'DOCX', 'DOC'])
keywords = st.text_area('Enter the keywords here...')
contents = []
scores = []

df = pd.DataFrame()

df = go_to_threads(files)
clean_keywords = filter_text(keywords)


if clean_keywords != '':
    for cv in df.iloc[:, 0].values.tolist():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thread = executor.submit(get_scores, cv, clean_keywords)
            result_value = thread.result()
            scores.append(result_value)
    
    df['scores'] = scores
    df['scores'] = df['scores'] * 100

    record = show(df[['bytes', 'title', 'scores', 'postition', 'phone_number', 'email']].sort_values(by='scores', ascending=False).drop_duplicates())
    
    new_phone_number = ''
    row = ''
    try:
        row = df[df['title']==record['title']]
        print(row)
        new_phone_number = st.text_input("Is this the phone number that you want to send the message to ?", row['phone_number'].values[0])
    except:
            pass


    
    try:
        for file in files:
            if file.name == record['title']:
                st.download_button(
                    label="Download file",
                    data=bytes(file.getbuffer()),
                    file_name=row['title'].values[0],
                )
    except:
        pass


    st.write('--------------------------------------------------------------')
    st.header('Chatbot')
    
    years_of_experience = st.checkbox('What is the total years of experience ?')
    nationality = st.checkbox('What is the nationality ?')
    work_experience = st.checkbox('What is the work experience ?')
    skills = st.checkbox('What are the skills ?')
    technical_skills = st.checkbox('What are the technical skills ?')
    visa_type = st.checkbox('What is the visa type if it\'s exist (visit / residence) ?')
    Custom = st.checkbox('Custom question')
    
    questions_list = []
    if years_of_experience:
        questions_list.append('What is the total years of experience ?')
    if nationality:
        questions_list.append('What is the nationality ?')
    if work_experience:
        questions_list.append('What is the work experience ?')
    if skills:
        questions_list.append('What is the skills ?')
    if technical_skills:
        questions_list.append('What is the technical skills ?')
    if visa_type:
        questions_list.append('What is the visa type if it\'s exist (visit / residence) ?')
        
    if Custom:
        question = st.text_input('Ask A Question...')
        questions_list.append(question)

    # res = ''
    # try:
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f'''
            answer the user question from this context as points and highlight the important information and be direct in your response
            context: {' '.join(row['cv'].values)}
            question: {' '.join(questions_list)}
            don't add extra information just answer the question
        ''',
        max_tokens=300,
        temperature=0
    )
    res = response.choices[0].text
    # except:
    #     pass
    
    if st.button('ask'):
        st.write('The answer is: ', res)
        # if st.button('read the answer'):
        #     read_text(response.choices[0].text)
            
            
        #     # txt2speech.convert(text=response.choices[0].text)
        #     with open('audio.wav', 'rb') as audio_file:
        #         audio_bytes = audio_file.read()
        #     print('show audio...')
        #     st.audio(audio_bytes, format='audio/wav')
            
            # audio = text_to_audio(response.choices[0].text,language='en')
            # auto_play(audio)


    st.write('---------------------------------------------')
    st.bar_chart(df['postition'].value_counts())

    fig = px.pie(df, names='postition', title='Positions Percentage')
    st.write('---------------------------------------------')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error('Enter Some Keywords...')
