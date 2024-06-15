import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("books_data.csv")
data.head()
data.columns
data.info()
# Convert 'average_rating' to a numeric data type
data['average_rating'] = pd.to_numeric(data['average_rating'], 
                                       errors='coerce')
# Create a new column 'book_content' by combining 'title' and 'authors'
data['book_content'] = data['title'] + ' ' + data['authors']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])

def recommend_books(book_title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = data[data['title'] == book_title].index[0]

    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended books
    return data['title'].iloc[book_indices]

book_title = input("enter the bookane:")
recommended_books = recommend_books(book_title)
print(recommended_books)

# Create GUI window

import tkinter as tk
from tkinter import simpledialog

window = tk.Tk()
window.title("Book Recommendation System")

# Input dialog
input_label = tk.Label(window, text="Enter an Book name:")
input_label.pack()

recommend_button = tk.Button(window, text="Get Recommendations", command=recommedations)
recommend_button.pack()

output_text = tk.Text(window, height=10, width=50)
output_text.pack()

window.mainloop()