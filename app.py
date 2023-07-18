import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import textract
from summary_algorithm import summarize_text
import PyPDF2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    # Loop through each page and extract its contents
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text


def extract_text_from_docx(file_path):
    text = textract.process(file_path)
    return text.decode('utf-8')


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summary', methods=['POST'])
def generate_summary():
    uploaded_files = request.files.getlist('file')
    percentage = int(request.form.get('percentage'))
    algorithm = request.form.get('algorithm')
    algorithm_name = {
        'TfIdfPos': 'TF-IDF and POS-based Summarization',
        'TextRank': 'TextRank Summarization',
        'LSASumy': 'Latent Semantic Analysis Summarization',
    }[algorithm]

    summaries = []
    total_input_word_count = 0
    total_output_word_count = 0
    combined_summary_words = []

    for file in uploaded_files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text = extract_text_from_file(file_path)
            summary = summarize_text(text, percentage, algorithm)

            os.remove(file_path)  # Remove the uploaded file

            # Count the number of words in the input and output
            input_word_count = len(text.split())
            output_word_count = len(summary.split())

            summaries.append({
                'input_file_data': text,  # <-- Add the input file data to the dictionary
                'summary': summary,
                'input_word_count': input_word_count,
                'output_word_count': output_word_count
            })

            combined_summary_words.extend(summary.split())
            total_input_word_count += input_word_count
            total_output_word_count += output_word_count

    combined_summary = ' '.join(combined_summary_words)
    combined_output_word_count = len(combined_summary.split())  # Calculate the combined output word count

    return render_template('summary.html',
                           summaries=summaries,
                           total_input_word_count=total_input_word_count,
                           total_output_word_count=total_output_word_count,
                           combined_summary=combined_summary,
                           combined_output_word_count=combined_output_word_count,
                           algorithm_name=algorithm_name,
                           percentage=percentage)


if __name__ == '__main__':
    app.run(debug=True)



