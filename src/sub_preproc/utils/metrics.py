from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def calculate_bleu(text1, text2):
    """
    Calculate BLEU score between two texts.
    """

    if text1 is None or text2 is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[text1.split()],
            hypothesis=text2.split(),
            smoothing_function=chencherry.method4,
        )


def calculate_wer(text1, text2):
    """
    Calculate Word Error Rate between two texts.
    """
    try:
        return wer(text1, text2)
    except:
        return None


def calculate_cer(text1, text2):
    """
    Calculate Character Error Rate between two texts.
    """

    try:
        return cer(text1, text2)
    except:
        return None
