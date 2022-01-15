import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 
import re 
pd.set_option('display.max_columns',None)
import warnings
warnings.filterwarnings('ignore')



#@st.cache(suppress_st_warning=True) 
def load_data(data):
    df=pd.read_csv(data)
    return df


def general_recommendation(df_items, df_ratings, UserId_column, DataId_column, data_name_column, rating_column, content_column, user_inp, num_of_recom, extra_col):
    
    
    df_items['clean_plot'] = df_items[content_column].str.lower()
    df_items['clean_plot'] = df_items['clean_plot'].apply(lambda x:re.sub('[^a-zA-Z]',' ',x))
    df_items['clean_plot'] = df_items['clean_plot'].apply(lambda x:re.sub('\s+',' ',x))
    
    df_items = df_items[[DataId_column, data_name_column ,content_column,extra_col, 'clean_plot']]
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(df_items['clean_plot'])
    
    merged_data=df_ratings.merge(df_items, on=DataId_column,how='left')
    #keeping only those columns which are relevant
    df=merged_data[[UserId_column, DataId_column, rating_column, data_name_column, extra_col]]
    #taking the sample from the large dataset
    df1=df.copy()
    #reset the shuffled index
    #creating cosine similarity matrix 
    cosine_sim = cosine_similarity(features,features)
    
    index = pd.Series(df_items[data_name_column])
    
    #item based collaborative filtering
    item_features=df1.pivot_table(index=[data_name_column],columns=[UserId_column],values=rating_column)
    item_features.fillna(0,inplace=True)
    
    # extract receipe names into list
    #Creating a dictionary with receipe name as key and its index from the list as value:
    
    
    def recommend_items(user_inp):
        items = []
        idx = index[index == user_inp].index[0]
        #print(idx)
        score = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        top10 = (score.iloc[1:num_of_recom].index)
        items=df_items[[DataId_column,data_name_column, content_column, extra_col]].iloc[top10]
        return items
    return recommend_items(user_inp)
    
    

    
def main():
    
    st.title("Welcome to Recommender System")
    
    menu=['Recommend','About']
    choice=st.sidebar.selectbox("Menu",menu)
    
    df1=load_data('movie_titles.csv')
    df2=load_data('Receipes_Titles.csv')
    df3=load_data('Books_name.csv')
    df1_ratings=load_data('ratings_movie.csv')
    df2_ratings=load_data('User Data Receipes.csv')
    df3_ratings=load_data('Ratings_Books.csv')
            
    if choice=='Recommend':
        st.subheader("Recommend Me")
        image1=Image.open('m2.jpg')
        image2=Image.open('b3.jfif')
        image3=Image.open('r3.jpeg')
        st.image([image1,image3,image2],width=220)
        #search_item= st.select_box("Search")
        #num_of_recom= st.sidebar.number_input('Number',4,20,5)
        select_data=st.sidebar.selectbox('What do you want to be recommended?'\
                                        ,options=['Movies','Receipes',\
                                               'Books'])
        num_recom=st.sidebar.slider("Number", min_value=4, max_value=20,\
                                   step=1)
        
        if select_data=='Movies':
            search_item=st.selectbox("Search your Movie",df1['title'])
            if st.button("Recommend"):
                result= general_recommendation(df1, df1_ratings, 'userId', 'movieId', 'title', 'rating', 'genres', search_item, num_recom, 'movie_url')
                st.write(result)
        elif select_data=='Receipes':
            search_item=st.selectbox("Search your Receipe",df2['name'])
            if st.button("Recommend"):
                result= general_recommendation(df2, df2_ratings, 'user_id', 'recipe_id','name','rating', 'ingredients', search_item, num_recom, 'steps')
                st.write(result)
        else:
            search_item=st.selectbox("Seach your Book",df3['title'])
            if st.button("Recommend"):
                result= general_recommendation(df3, df3_ratings, 'user_id', 'book_id','title' ,'rating', 'tag_name_y', search_item, num_recom, 'image_url')
                st.write(result)
                
        
    else:
        image = Image.open('r1.png')
        st.image(image, width=700)
        st.subheader("About")
        st.text("Built with streamlit and Pandas")
        
if __name__ == "__main__":
    main()

