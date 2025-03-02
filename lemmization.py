import pandas as pd
from camel_tools.data import CamelData
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

def initialize_analyzer():
    """
    Initialize the CAMeL Tools analyzer with error handling for missing database.
    
    Returns:
    --------
    Analyzer
        An initialized CAMeL Tools morphological analyzer.
    """
    try:
        db = MorphologyDB.builtin_db()
        return Analyzer(db)
    except FileNotFoundError:
        print("Error: Morphology database not found. Running download...")
        # Use the correct camel_tools data download method
        data = CamelData()
        data.download_morphology_db()
        print("Download complete. Initializing analyzer...")
        db = MorphologyDB.builtin_db()
        return Analyzer(db)

# Initialize the analyzer globally
analyzer = initialize_analyzer()

def lemmatize_arabic(text):
    """
    Lemmatize Arabic text using CAMeL Tools.
    
    Parameters:
    -----------
    text : str
        The Arabic text to lemmatize.
        
    Returns:
    --------
    str
        The lemmatized text with words joined by spaces.
    """
    # Handle empty or non-string input
    if not isinstance(text, str) or not text.strip():
        return text
    
    # Split into words
    words = text.split()
    
    # Get lemmas for each word
    lemmatized_words = []
    for word in words:
        try:
            analyses = analyzer.analyze(word)
            if analyses:
                # Get the lemma from the first analysis
                lemma = analyses[0]['lemma']
                lemmatized_words.append(lemma)
            else:
                # If no analysis is found, keep the original word
                lemmatized_words.append(word)
        except Exception as e:
            # Keep original word if any error occurs
            print(f"Error processing word '{word}': {str(e)}")
            lemmatized_words.append(word)
    
    # Join the lemmatized words with spaces
    return " ".join(lemmatized_words)

def lemmatize_dataframe_column(df, column_name, new_column_name=None):
    """
    Apply lemmatization to a specific column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the text to lemmatize.
    column_name : str
        The name of the column containing Arabic text.
    new_column_name : str, optional
        The name of the new column to store lemmatized text.
        If None, will use column_name + '_lemmatized'.
        
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the new lemmatized column added.
    """
    # Set the default new column name if not provided
    if new_column_name is None:
        new_column_name = f"{column_name}_lemmatized"
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply lemmatization to the specified column
    print(f"Lemmatizing column '{column_name}'...")
    result_df[new_column_name] = result_df[column_name].apply(lemmatize_arabic)
    print("Lemmatization complete!")
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        'id': [1, 2, 3],
        'arabic_text': [
            "الطلاب يذهبون إلى المدرسة",
            "القراءة مفيدة للعقل",
            "أكلت التفاحة بالأمس"
        ]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Lemmatize the 'arabic_text' column
    lemmatized_df = lemmatize_dataframe_column(df, 'arabic_text')
    
    # Display the results
    print("\nLemmatized DataFrame:")
    print(lemmatized_df)