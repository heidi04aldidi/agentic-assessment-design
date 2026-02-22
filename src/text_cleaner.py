# text_cleaner.py

# basic libs
import pandas as pd
import re
from bs4 import BeautifulSoup 


# 1) Load tags from the csv we created (unique_tags.csv)
def load_tags(tag_csv_path="../data/processed/unique_tags.csv"):
    # read the tags file
    tags_df = pd.read_csv(tag_csv_path, encoding="latin1")

    # convert column to clean list
    # remove NaN, make lowercase, strip spaces
    tags = (
        tags_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .tolist()
    )

    # we will temporarily replace tags with placeholders
    # example: python -> TAGTOKEN0
    tag_placeholder_map = {}

    for i, tag in enumerate(tags):
        tag_placeholder_map[tag] = f"TAGTOKEN{i}"

    # reverse map to bring tags back later
    reverse_map = {v: k for k, v in tag_placeholder_map.items()}

    return tag_placeholder_map, reverse_map



# 2) Remove HTML from stackoverflow posts
# (because Body column has <p> <code> <a> etc)

def remove_html(text):
    if pd.isna(text):
        return ""

    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text(separator=" ")


# 3) Protect tags BEFORE cleaning
# otherwise cleaning destroys stuff like c++, c#, node.js

def protect_tags(text, tag_placeholder_map):
    if pd.isna(text):
        return ""

    text = text.lower()

    # replace each tag with placeholder
    for tag, placeholder in tag_placeholder_map.items():
        text = re.sub(rf"\b{re.escape(tag)}\b", placeholder, text)

    return text


# 4) Actual text cleaning
# remove symbols, punctuation, weird chars

def clean_text(text):
    # keep letters and numbers only
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 5) Bring tags back after cleaning
# TAGTOKEN0 -> python

def restore_tags(text, reverse_map):
    for placeholder, tag in reverse_map.items():
        text = text.replace(placeholder, tag)

    return text


# 6) THE MAIN FUNCTION (this is what teammates will use)
# full cleaning pipeline

def cleanText(text, tag_placeholder_map, reverse_map):

    # step 1 remove html
    text = remove_html(text)

    # step 2 protect programming tags
    text = protect_tags(text, tag_placeholder_map)

    # step 3 normal cleaning
    text = clean_text(text)

    # step 4 restore tags back
    text = restore_tags(text, reverse_map)

    return text


# 7) Apply to whole dataframe column
# works for BOTH Questions and Answers dataset

def clean_dataframe(df, column_name, tag_csv_path="../data/processed/unique_tags.csv"):

    # load tags + placeholder mapping
    tag_placeholder_map, reverse_map = load_tags(tag_csv_path)

    # create new cleaned column
    # example: Body -> Body_cleaned
    df[column_name + "_cleaned"] = df[column_name].apply(
        lambda x: cleanText(x, tag_placeholder_map, reverse_map)
    )

    return df