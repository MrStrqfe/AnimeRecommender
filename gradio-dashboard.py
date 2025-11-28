import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# Load data and environment
load_dotenv()

animes = pd.read_csv("animes_with_emotions.csv")

# Fix image paths
animes["image"] = animes["image"].fillna("anime_not_found.jpg")

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_animes = Chroma.from_documents(documents, OpenAIEmbeddings())

# Recommendation Retrieval
def retrieve_semantic_recommendations(
        query: str,
        genre: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # similarity search
    recs = db_animes.similarity_search(query, k=initial_top_k)

    animes_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    anime_recs = animes[animes["myanimelist_id"].isin(animes_list)].head(final_top_k)

    if genre != "All":
        anime_recs = anime_recs[anime_recs["simple_genres"] == genre][:final_top_k]
    else:
        anime_recs = anime_recs.head(final_top_k)


    if tone == "Happy":
        anime_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        anime_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        anime_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        anime_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        anime_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return anime_recs

def recommend_animes(
        query: str,
        genre: str,
        tone: str,
):

    recommendations = retrieve_semantic_recommendations(query, genre, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        studios_split = row["Studios"].split(",")
        if len(studios_split) == 2:
            studios = f"{studios_split[0]} {studios_split[1]}"
        elif len(studios_split) > 2:
            studios_str = f"{', '.join(studios_split[:1])}, and {studios_split[-1]}"
        else:
            studios_str = row["Studios"]

        caption = f"{row['title']} by {studios_str}: {truncated_description}"
        results.append((row["image"], caption))
    return results

genres = ["All"] + sorted(animes["simple_genres"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Anime Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of the anime:",
                                placeholder = "e.g., A anime about survival")
        genre_dropdown = gr.Dropdown(choices = genres, label = "Select a genre:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended Animes", columns = 8, rows = 2)

    submit_button.click(fn = recommend_animes,
                        inputs = [user_query, genre_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())