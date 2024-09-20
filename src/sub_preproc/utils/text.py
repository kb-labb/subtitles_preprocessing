import re
import string
import unicodedata

from num2words import num2words

from sub_preproc.utils.strings_sv import abbreviations, ocr_corrections, symbols


def expand_abbreviations(text):
    """
    Replace abbreviations with their full form in the text.
    """

    # Normalize variations of abbreviations
    for key, value in ocr_corrections.items():
        text = text.replace(key, value)

    # Replace abbreviations with their full form
    for key, value in abbreviations.items():
        text = text.replace(key, value)

    # Replace symbols with their full form
    for key, value in symbols.items():
        text = text.replace(key, value)

    return text


def normalize_text(text):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text.

    Args:
        text (str): The text to normalize.
    Returns:
        str: The normalized text.
    """

    text = text.lower()
    text = expand_abbreviations(text)  # Replace abbreviations with their full form

    # # Convert numbers to words
    text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Remove multiple spaces and replace with single space
    text = re.sub(r"\s+", " ", text)
    ## Remove whitespace between numbers
    # text = re.sub(r"(?<=\d) (?=\d)", "", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def normalize_text_svt(text):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text.

    Args:
        text (str): The text to normalize.
    Returns:
        str: The normalized text.
    """
    text = text.lower()
    # Replace abbreviations with their full form
    text = expand_abbreviations(text)
    # Remove hyphens
    text = re.sub(r"(?<=\s)–\b", "", text)
    # Remove hyphens between words
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"(?<=\w)–(?=\w)", " ", text)
    # Remove "/  /"  and everything between them
    text = re.sub(r"/.*?/", " ", text)
    # Convert numbers to words
    text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Remove newlines in the text
    text = text.replace("\n", " ")
    # Remove "-" and "–" in the beginning of the text
    text = text.lstrip("-")
    text = text.lstrip("–")
    # Remove multiple spaces and replace with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def clean_subtitle(text):
    """
    Cleaning subtitle before outputting final ground truth to text files.
    """
    text = re.sub(r"\s+", " ", text)  # Youtube has newlines in the text
    # To handle: "när man har hittat sitt drömhus– –vilken strategi ska jag ha"
    text = text.replace("- -", " ")
    text = text.replace("– –", " ")
    return text
