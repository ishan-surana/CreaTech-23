# Sentiment analysis model for CreaTech '24
### Team - *I&T Solutions*

An AI model created for the CreaTech 2024 hackathon, on the problem statement:-<br>
`"Develop an AI model to understand the sentiments about the company using social media feeds (such as Twitter, Facebook, LinkedIn, and other digital media inputs)."`

Hosted [here](https://i-and-t-solutions-createch-24.streamlit.app/) via Streamlit.

> [!NOTE]
> Above link is *old* model (initial submission). Revamped model hosted **[here](https://createch-24-i-and-t-solutions.streamlit.app/)**.

## Description
The [dataset](reddit_posts_data.csv) used in the model has been formed by scraping data from Reddit. The codes have been made completely by me and use DOM to access the textarea of the posts and links. The [scaper](scraper.py) executes the scripts provided to access relavent post data and stores them in the dataset. The posts were extracted on the search phrases "larsen and toubro" and "l&t".

The [model](model.py) does the following steps:-
+ *Data preprocessing* = The data is cleaned by removing excess spaces and lemmatized. Part-of-speech tagging is employed and text is tokenized. Then the stopwords are removed (utilised nltk), and finally, structural features of the text are extracted.
+ *Label mapping and upvote normalized* = The labels (negative, neutral, and negative) are mapped to (0,1,2) for numeric calculations. The upvotes are normalized to bring values from 0 to 1 in order to account for magnitude.
+ *Model architecture* = The components and layers of the model and their purposes are as follows:-
  - Defined input layers for textual content and upvotes.
  - Embedding layer to convert input sequences into dense vectors.
  - Convolutional layer followed by global max pooling for extracting features from textual content.
  - Dense layer for processing upvotes.
  - Concatenated the outputs from the convolutional and dense layers.
  - Additional fully connected layers for further processing.
  - Output layer with softmax activation for multi-class classification.
+ *Testing* = A while loop established for user to test data by inputting post data and upvotes. In response, it will be processed by the model and the model will determine the magnitudes of the post being positive, negative or neutral.