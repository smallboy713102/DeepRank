# Resume Classification and Ranking System

## Overview
This project is a Resume Classification and Ranking System that processes multiple resumes in PDF format, classifies them into predefined job categories, and ranks them based on relevance to a specified job role.

## Features
- Extracts text from resumes in PDF format.
- Cleans and preprocesses text using regex.
- Tokenizes and sequences text for model input.
- Uses a deep learning model (CNN + LSTM) to classify resumes.
- Ranks resumes based on their softmax probability score for a given job role.
- Normalizes scores and sorts resumes in descending order of relevance.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas tensorflow scikit-learn PyPDF2
```

## Dataset
The model is trained using the `UpdatedResumeDataSet.csv`, which contains resumes and their corresponding job categories.

## Model Architecture
- **Embedding Layer**: Converts words into dense vectors.
- **Conv1D Layer**: Captures local dependencies in text.
- **MaxPooling1D Layer**: Reduces dimensionality.
- **LSTM Layer**: Extracts long-term dependencies.
- **Dropout Layer**: Prevents overfitting.
- **Dense Layer with Softmax Activation**: Outputs probability distribution across job categories.

## Usage
1. Place resumes in the `Resumes` folder.
2. Load the pre-trained model weights (`deeprank_model.h5`).
3. Run the script to classify and rank resumes for a given job role.
4. Example:
```python
pdf_folder = "Resumes"
job_role = "Data Science"
ranked_resumes = process_resumes(pdf_folder, job_role)
print(ranked_resumes)
```

## Output
The script returns a sorted DataFrame containing:
- Resume file name
- Job role probability score
- Normalized score (0-1 scale)

## License
This project is licensed under the MIT License.

## Author
Nilesh Ranjan Pal

