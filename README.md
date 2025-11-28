# Anime Recommender using LLMs
An intelligent anime recommendation system powered by Large Language Models, 
semantic search, and emotion-aware ranking. Inspired by 
freeCodeCamp’s Build a Semantic Book Recommender.

## ✨ Features
* Semantic search using OpenAI embeddings + ChromaDB
* Emotion-based recommendations (joy, sadness, anger, fear, surprise)
* Genre filtering for targeted results
* Interactive Gradio dashboard UI
* Fully local vector database created from anime descriptions
* Built with Python 3.13

## 📦 Dependencies
Project is created in Python 3.13. In order to run the project, the following dependencies are required:
* [kagglehub](https://pypi.org/project/kagglehub/)
* [pandas](https://pypi.org/project/pandas/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [seaborn](https://pypi.org/project/seaborn/)
* [python-dotenv](https://pypi.org/project/python-dotenv/)
* [langchain-community](https://pypi.org/project/langchain-community/)
* [langchain-opencv](https://pypi.org/project/langchain-opencv/)
* [langchain-chroma](https://pypi.org/project/langchain-chroma/)
* [transformers](https://pypi.org/project/transformers/)
* [gradio](https://pypi.org/project/gradio/)
* [notebook](https://pypi.org/project/notebook/)
* [ipywidgets](https://pypi.org/project/ipywidgets/)

## 🔑 Environment Setup
Create a `.env` file in the project root:
```OPENAI_API_KEY=your_api_key_here```

## 📁 Dataset
`animes_with_emotions.csv`
Also ensure you have:
`anime_not_found.jpg` (fallback image)

## 🧠 How It Works
1. Anime descriptions are loaded and split using LangChain’s text splitter.
2. Each chunk is embedded using OpenAI embeddings.
3. A Chroma vector database is created from those embeddings.
4. When the user submits a query, the system performs semantic similarity search.
5. Matching anime IDs are filtered and ranked according to:
6. Genre (optional)
7. Emotional tone (optional)
8. The top results are displayed in a Gradio gallery with images and short descriptions.

## 🚀 Running the Project
Launch the Gradio dashboard
`python or python3 gradio-dashboard.py`
The app will run at:
`http://127.0.0.1:7860`

## 🖥️ Project Structure
```.
├── gradio-dashboard.py
├── animes_with_emotions.csv
├── tagged_description.txt
├── anime_not_found.jpg
├── .env
└── README.md
```

## 📚 Credits
* Inspired by freeCodeCamp's semantic search projects
* Built with LangChain, ChromaDB, and OpenAI embeddings
* Dataset from Kaggle

