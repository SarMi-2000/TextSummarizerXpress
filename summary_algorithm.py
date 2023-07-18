import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def summarize_text(text, percentage, algorithm):
    if algorithm == 'TfIdfPos':
        return TfIdfPos(text, percentage)
    elif algorithm == 'TextRank':
        return TextRank(text, percentage)
    elif algorithm == 'LSASumy':
        return LSASumy(text, percentage)
    else:
        raise ValueError("Invalid algorithm choice")

def TfIdfPos(text, percentage):
    # Tokenizing the sentences
    sentences = sent_tokenize(text)
    cleaned_sentences = [re.sub(r'[^\w\s]', '', sentence).lower() for sentence in sentences]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_vectorizer.fit(cleaned_sentences)
    tagged_sentences = [nltk.pos_tag(word_tokenize(sentence)) for sentence in cleaned_sentences]
    sentence_scores = {}
    total_word_count = sum(len(sentence.split()) for sentence in sentences)

    # Calculate the number of words to include in the summary based on the selected percentage
    num_words = int((percentage / 100) * total_word_count)

    for i, sentence in enumerate(cleaned_sentences):
        sentence_tokens = word_tokenize(sentence)
        sentence_tfidf_scores = [tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_.get(token, 0)]
                                 for token in sentence_tokens]
        pos_scores = [1 if tag.startswith('NN') or tag.startswith('VB') else 0
                      for _, tag in tagged_sentences[i]]
        sentence_scores[sentence] = sum(sentence_tfidf_scores) * sum(pos_scores)

    # Select the most important sentences for the summary based on the number of words
    summary_sentences = []
    current_word_count = 0
    for sentence in sentences:
        if current_word_count + len(sentence.split()) <= num_words:
            summary_sentences.append(sentence)
            current_word_count += len(sentence.split())
        else:
            break

    summary = ' '.join(summary_sentences)

    return summary

def TextRank(text, percentage):
    # Create parser and tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Calculate the number of sentences to include in the summary based on the selected percentage
    num_sentences = int((percentage / 100) * len(parser.document.sentences))

    # Create summarizer
    summarizer = TextRankSummarizer()

    # Get summary and join the sentences
    summary_sentences = summarizer(parser.document, num_sentences)
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    return summary

def LSASumy(text, percentage):
    # Create parser and tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Calculate the number of sentences to include in the summary based on the selected percentage
    num_sentences = int((percentage / 100) * len(parser.document.sentences))

    # Create summarizer
    summarizer = LsaSummarizer()

    # Get summary and join the sentences
    summary_sentences = summarizer(parser.document, num_sentences)
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    return summary




