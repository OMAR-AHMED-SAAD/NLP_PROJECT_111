# NLP_PROJECT_111
## Goal Task

Our Goal Task is to train a Deep Learning Model for `Multi Label Extraction`

## Data Understanding

1. **Datasets Used**
   - Da7ee7
   - Al Mokhbir Al Eqtisadi
   - Fi Al Hadaraa
2. **Data Loading**
   - Loaded the `.txt` files into a pandas dataframe
   - Extracted the length & tags from the metadata files and loaded it into a pandas dataframe

3. **Basic Analysis**
   - Check Missing Values 
   - Check Duplicates
   - Check datatypes
   - Check basic statisitcs using the `.describe()` method

3. **Length & Transcript Length Analysis**
   - View the mean episode length per creator
   - Plot the distribution (histogram) for the length per creator
   - View the mean transcript length per creator
   - Plot the distribution (histogram) for the transcript length per creator
   - View the correlation between the length and the transcript length
   - Plot the scatter plot between the length and the transcript length

4. **Non Arabic Word Analysis**
   - Count the words using the English Alphabet in each Transcript using Regular Expressions
   - Sum the number of Non Arabic Words per Creator
   - Removed a fully English Episode in `Da7ee7` dataset, which was probably a translated version as the original episode on youtube was in Arabic
   - Plot the histogram distribution of the usage of Non Arabic Words per creator
   - Plot the Word Clouds for the Non Arabic Words for each creator.
   - Analyse the Word Clouds for each creator

5. **Arabic Word Analysis**
   - Plot the Word Clouds for the Arabic Words for each creator.
   - Plot the Phrase Cloud for the Arabic transcripts for each creator:
     - **Clean and tokenize text**:
       - Converts text to lowercase.
       - Removes punctuation.
       - Splits text into tokens (words).
     - **Generates n-grams**: (bigrams, trigrams, etc.) from the tokenized text.
     - **Plot the phrase cloud using different n-grams**
   - Visualizes the **top K most frequent n-grams** by counting the occurrences of each n-gram (bigrams, trigrams, etc.) for each creator.

6. **Sentiment Analysis**
   - Use the [Arabic Sentiment Analysis model](https://huggingface.co/Walid-Ahmed/arabic-sentiment-model) to obtain the sentiment of each transcript
   - Plot the distribution (histogram) of Sentiment for every creator
   - Draw the boxplot for the sentiment for each creator to compare the spread, Quartiles, Min & Max of each sentiment

7. **Sarcasm Analysis**
   - Use the [Arabert Sarcasm Detector](https://huggingface.co/MohamedGalal/arabert-sarcasm-detector) to obtain if each transcript is sarcastic or not
   - Use [Gemini API](https://deepmind.google/technologies/gemini/flash/) to if each transcript is sarcastic or not
   - Plot a bar plot for each creator to view how many transcripts have sarcasm and how many don't

8. **Named Entity Recognition (NER)**
   - Use the [Marefa Arabic Named Entity Recognition Model](https://huggingface.co/marefa-nlp/marefa-ner) to extract the entities.
   - Filter Unique Entities and plot the counts per creator.
   - Check the intersection in Entities between the different creators.

9.  **TF-IDF & Clustering**
   - Clustering sample of episodes based on TF-IDF vectors of the transcript to find if the data is usable for clustering or not.
   - Visualize the episodes using UMAP (2D and 3D)
   - Use KMeans to cluster the reduced embeddings.
   - Visualize the clusters in 3D.



## Preprocessing Steps

1. **Remove Timestampes**
   - such as (4.25)
2. **Remove Tags**
   - such as [موسيقي]
3. **Remove Ellipses**
4. **Seperate punctuations & quotations from words**
5. **Remove all punctuantions**
6. **Remove english characters**
7. **Remove Diacritics (Tashkeel)**
8. **Remove Elongation (Tatweel)**
9. **Remove stopwords**

## Obtaining Labels
1. **Using TF-IDF**
   - Get the most repeated words to act as labels
2. **Using Gemini**
   - Give the transcript for gemini to get multiple labels that can be used