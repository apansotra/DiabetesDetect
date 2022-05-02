

import pickle
#from flask import Flask, render_template, request
import streamlit as st
import pandas as pd
import numpy as np

#create flask app
app = Flask(__name__)
#load pickle model
model=pickle.load(open("model.pkl","rb"))


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':

        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        
        input_variables = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],
                                       columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        def res():
            if prediction == 0 :
                return "No"
            elif prediction == 1:
                return "Yes"
        #print(prediction)
        return render_template('main.html',
                                     original_input={'Pregnancies':Pregnancies,
                                                     'Glucose':Glucose,
                                                     'BloodPressure':BloodPressure,
                                                     'SkinThickness':SkinThickness,
                                                     'Insulin':Insulin,
                                                     'BMI':BMI,
                                                     'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                                                     'Age':Age},
                                     result = res(),
                                     )




if __name__=="__main__":
       app.run(host='localhost', port=5002,debug=True)


import streamlit as st 
import pickle
import pandas as pd 
import requests
import urllib.request


def fetch_poster(movie_id):
    respose = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=88cacc33658425acdb11fa37422eb4e2&language=en-US'.format(movie_id)).json()
    return "https://image.tmdb.org/t/p/w500/" + respose['poster_path']

def recommand(movie):
    new_df = movies
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    recommand_movies= []
    recommanded_poster = []
    for i in movies_list:
        recommand_movies.append(new_df.iloc[i[0]].title)
        #fetch poster from TMDB API
        recommanded_poster.append(fetch_poster(new_df.iloc[i[0]].id))
        

    return recommand_movies,recommanded_poster

# similarity = pickle.load(open('similarity.pkl','rb'))
# movie_dict = pickle.load(open('movie_dict.pkl','rb'))

#we use google drive api for large file
movie_dict =  pickle.load(urllib.request.urlopen(("https://www.googleapis.com/drive/v3/files/1nJmfC1VBVTHM5bo6hsChVXazWpU8d8tN?alt=media&key=AIzaSyCUpIPtvM6lJw65WnZoM-Xxn7Xo6xytJ3k")))
similarity =  pickle.load(urllib.request.urlopen(("https://www.googleapis.com/drive/v3/files/1PGoyJzpTSkosynS8Rg-X-wTGXXnIq7fH?alt=media&key=AIzaSyCUpIPtvM6lJw65WnZoM-Xxn7Xo6xytJ3k")))



st.title('Diabetes Prediction System')



selected_movie = st.selectbox('Select Movie ',movies['title'].values)

if st.button('recommand'):
    names,poster = recommand(selected_movie)

    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(poster[0])
    with col2:
        st.text(names[1])
        st.image(poster[1])
    with col3:
        st.text(names[2])
        st.image(poster[2])
    with col4:
        st.text(names[3])
        st.image(poster[3])
    with col5:
        st.text(names[4])
        st.image(poster[4])
