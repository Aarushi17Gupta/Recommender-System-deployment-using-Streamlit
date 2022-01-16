import streamlit as st
#import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer 
import re 
pd.set_option('display.max_columns',None)
import warnings
warnings.filterwarnings('ignore')




def load_data(data):
    df=pd.read_csv(data)
    return df

def general_recommendation(df_items, DataId_column, data_name_column, rating_column, content_column, user_inp, num_of_recom, extra_col):
    

    df_items['clean_plot'] = df_items[content_column].str.lower()
    df_items['clean_plot'] = df_items['clean_plot'].apply(lambda x:re.sub('[^a-zA-Z]',' ',x))
    df_items['clean_plot'] = df_items['clean_plot'].apply(lambda x:re.sub('\s+',' ',x))
    
    df_items = df_items[[DataId_column, data_name_column ,content_column, extra_col,rating_column ,'clean_plot']]
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(df_items['clean_plot'])
    
    cosine_sim = linear_kernel(features,features)
    
    index = pd.Series(df_items[data_name_column])
    
    
    def recommend_items(name=user_inp):
        items = []
        idx = index[index == name].index[0]
        #print(idx)
        score = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        top10 = (score.iloc[1:num_of_recom].index)
        items=df_items[[DataId_column,data_name_column,content_column, extra_col,rating_column]].iloc[top10]
        final=items.sort_values(by=rating_column,ascending=False)
        return final.reset_index(drop=True)
    return recommend_items(name=user_inp)


def main():
    
    st.title("Welcome to Recommender System")
    
    menu=['Recommend','About']
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice=='Recommend':
        st.subheader("Recommend Me")
        image1=Image.open('m2.jpg')
        image2=Image.open('b3.jfif')
        image3=Image.open('r3.jpeg')
        image4=Image.open('O1.jpg')
        st.image([image1,image3,image2,image4],width=220)
        
        select_data=st.sidebar.selectbox('What do you want to be recommended?'\
                                        ,options=['Movies','Receipes',\
                                               'Books','MOOCs'])
        num_recom=st.sidebar.slider("Number", min_value=4, max_value=20,\
                                   step=1)
        
        if select_data=='Movies':
            df1=load_data('MOVIES.csv')
            #df1_ratings=load_data('ratings_movie.csv')
            search_item=st.selectbox("Search your Movie",df1['title'])
            if st.button("Recommend"):
                result= general_recommendation(df1, 'movieId', 'title', 'rating', 'genres', search_item, num_recom, 'movie_url')
                st.write(result)
        elif select_data=='Receipes':
            df2=load_data('RECEIPES.csv')
            #df2_ratings=load_data('ratings_Receipes.csv')
            search_item=st.selectbox("Search your Receipe",df2['name'])
            if st.button("Recommend"):
                result= general_recommendation(df2, 'recipe_id','name','rating', 'ingredients', search_item, num_recom, 'steps')
                st.write(result)
        elif select_data=='Books':
            df3=load_data('BOOKS.csv')
            #df3_ratings=load_data('Ratings_Books.csv')
            search_item=st.selectbox("Seach your Book",df3['title'])
            if st.button("Recommend"):
                result= general_recommendation(df3, 'book_id','title' ,'rating', 'tag_name_y', search_item, num_recom, 'image_url')
                st.write(result)
        else:
            df4=load_data('MOOCs.csv')
            df4['tags']=df4['tags'].apply(str)
            search_item=st.selectbox("Seach your Online Course",df4['title'])
            if st.button("Recommend"):
                result=general_recommendation(df4, 'course_id', 'title', 'Rating', 'tags', search_item, num_recom, 'course_url')
                st.write(result)
            
                
        
    else:
        image = Image.open('r1.png')
        st.image(image, width=700)
        st.subheader("About")
        st.text(" This is a ML based general recommender system built with Pandas and Streamlit.\n It can recommend you Books, Movies and Receipes.\n By:- Aarushi Gupta")
        
if __name__ == "__main__":
    main()

