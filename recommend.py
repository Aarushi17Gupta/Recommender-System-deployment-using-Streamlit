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



@st.cache(max_entries=10, ttl=3600)
def load_data(data):
    df=pd.read_csv(data)
    return df


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'



def general_recommendation(item_dataset, user_dataset, UserId_column, DataId_column, data_name_column, rating_column, content_column, user_inp, num_of_recom, extra_col):
    df_items=pd.read_csv(item_dataset,header=0)
    df_ratings=pd.read_csv(user_dataset,header=0)
    
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
    
    
    def recommend_items(name=user_inp):
        items = []
        idx = index[index == name].index[0]
        #print(idx)
        score = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
        top10 = (score.iloc[1:num_of_recom].index)
        items=df_items[data_name_column].iloc[top10]
        return set(items)
    
    # Load Reader library
    reader = Reader()

    # Load ratings dataset with Dataset library
    data = Dataset.load_from_df(df_ratings[[UserId_column, DataId_column, rating_column]], reader)
    trainset = data.build_full_trainset()
    svd=SVD()
    svd.fit(trainset)
    
    def hybrid_content_svd_model(user_inp):

    #hydrid the functionality of content based and svd based model to recommend user top 10 movies. 
    #:param userId: userId of user
    #:return: list of movies recommended with rating given by svd model
        
        recommended_items_by_content_model = recommend_items(user_inp)
        recommended_items_by_content_model = df_items[df_items.apply(lambda item: item[data_name_column] in recommended_items_by_content_model, axis=1)]
        userId= (df_ratings[UserId_column][df_ratings[DataId_column]==int(df_items[DataId_column][df_items[data_name_column]== user_inp].values)]).values[0]
        for key, columns in recommended_items_by_content_model.iterrows():
            predict = svd.predict(userId, columns[DataId_column])
            recommended_items_by_content_model.loc[key, "svd_rating"] = predict.est
            if(predict.est < 2):
                recommended_items_by_content_model = recommended_items_by_content_model.drop([key])
            return (recommended_items_by_content_model.sort_values("svd_rating", ascending=False).iloc[0:num_of_recom].drop(['clean_plot'],axis=1)).reset_index(drop=True)
    return hybrid_content_svd_model(user_inp)

    
def main():
    
    st.title("Welcome to Recommender System")
    
    menu=["Home",'Recommend','About']
    choice=st.sidebar.selectbox("Menu",menu)
    
    df1=load_data('movie_titles.csv')
    df2=load_data('Receipes_Titles.csv')
    df3=load_data('Books_name.csv')
    if choice=='Home':
        st.subheader("Home")
        select_data=st.sidebar.selectbox('Dataframes for each category'\
                                        ,options=['Movies','Receipes',\
                                               'Books'])
        if select_data=='Movies':
            image = Image.open('m1.jpg')
            st.image(image, width=400)
            st.dataframe(df1)
            #df1.style.format({'movie_url': make_clickable})
        elif select_data=='Receipes':
            image = Image.open('r2.jpg')
            st.image(image, width=400)
            st.dataframe(df2)
        else:
            image = Image.open('b1.jpeg')
            st.image(image, width=400)
            st.dataframe(df3)
            
            
        
    
    elif choice=='Recommend':
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
                result= general_recommendation('movie_titles.csv', 'ratings_movie.csv', 'userId', 'movieId', 'title', 'rating', 'genres', search_item, num_recom, 'movie_url')
                st.write(result)
        elif select_data=='Receipes':
            search_item=st.selectbox("Search your Receipe",df2['name'])
            if st.button("Recommend"):
                result= general_recommendation("Receipes_Titles.csv",'User Data Receipes.csv', 'user_id', 'recipe_id','name','rating', 'ingredients', search_item, num_recom, 'steps')
                st.write(result)
        else:
            search_item=st.selectbox("Seach your Book",df3['title'])
            if st.button("Recommend"):
                result= general_recommendation("Books_name.csv",'Ratings_Books.csv', 'user_id', 'book_id','title' ,'rating', 'tag_name_y', search_item, num_recom, 'image_url')
                st.write(result)
                
        
    else:
        image = Image.open('r1.png')
        st.image(image, width=700)
        st.subheader("About")
        st.text("Built with streamlit and Pandas")
        
if __name__ == "__main__":
    main()

