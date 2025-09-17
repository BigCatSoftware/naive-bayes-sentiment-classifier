# Naive Bayes Sentiment Classifier

Statistical text classification system implementing Naive Bayes with Laplace smoothing for sentiment analysis of movie reviews.

## Overview

This project implements a probabilistic text classifier using the Naive Bayes algorithm with Laplace smoothing for binary sentiment classification. The system processes natural language text, extracts meaningful features through tokenization and stopword removal, and applies statistical methods to classify sentiment with quantifiable confidence scores.

## Key Features

- **Probabilistic Classification**: Implements maximum likelihood estimation with add-1 smoothing
- **Natural Language Processing**: NLTK-based tokenization, stopword removal, and optional stemming
- **Statistical Validation**: Comprehensive accuracy metrics and error analysis
- **Feature Engineering**: Word frequency analysis and vocabulary extraction
- **Performance Analysis**: Detailed misclassification reporting and confidence scoring
- **Batch Processing**: Automated sentiment prediction for large text collections

## Mathematical Foundation

The classifier implements Naive Bayes using log-likelihood ratios to prevent numerical underflow:

**Training Phase:**
- Prior probabilities: P(class) = count(class) / total_documents
- Likelihood estimation: P(word|class) = (count(word,class) + 1) / (total_words_in_class + vocabulary_size)
- Log-likelihood ratio: log(P(word|positive)) - log(P(word|negative))

**Classification Phase:**
- Decision score = log_prior + Σ log_likelihood[word] for words in document
- Classification rule: Positive if score > 0, Negative otherwise

**Laplace Smoothing** prevents zero probabilities for unseen words during training, ensuring robust classification of new text.

## Applications

- **Text Analytics**: Automated sentiment monitoring for customer feedback
- **Content Analysis**: Large-scale opinion mining and trend analysis  
- **Quality Assurance**: Statistical validation of classification algorithms
- **Research**: Baseline implementation for comparative NLP studies

## Performance Metrics

The system provides comprehensive evaluation including:
- **Classification Accuracy**: Overall correctness rate
- **Confidence Scoring**: Absolute decision boundary distances
- **Error Analysis**: Detailed misclassification examination
- **Statistical Summary**: Class distribution and prediction confidence

## Usage

### Basic Classification
```python
# Load and prepare training data
pos_reviews = load_reviews_from_folder('pos/pos', 1)
neg_reviews = load_reviews_from_folder('neg/neg', 0)

# Train classifier
word_counts = count_reviews(train_reviews, use_stemming=False)
logprior, loglikelihood = train_naive_bayes(word_counts, train_reviews)

# Classify new text
score = predict_naive_bayes("This movie was fantastic!", logprior, loglikelihood)
sentiment = "Positive" if score > 0 else "Negative"
```

### Command Line Execution
```bash
python Naive_SA.py
```

### Batch Processing
```bash
# Place reviews in LionKing_MovieReviews.txt (one per line)
# Run classifier to generate LionKing_Output.txt
```

## Dataset Structure

```
project/
├── pos/pos/          # Positive review text files
├── neg/neg/          # Negative review text files  
├── LionKing_MovieReviews.txt    # Sample reviews for prediction
├── stopwords.txt     # Custom stopwords list (optional)
└── Naive_SA.py       # Main implementation
```

## Dependencies

```python
nltk>=3.6
```

The system uses NLTK for natural language processing tasks including tokenization and stopword removal.

## Installation

1. **Clone repository:**
```bash
git clone https://github.com/BigCatSoftware/naive-bayes-sentiment-classifier.git
cd naive-bayes-sentiment-classifier
```

2. **Install dependencies:**
```bash
pip install nltk
```

3. **Download NLTK resources:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. **Dataset included:**
   - Positive reviews are provided in `pos/pos/` directory
   - Negative reviews are provided in `neg/neg/` directory
   - Sample reviews for testing in `LionKing_MovieReviews.txt`

5. **Run classifier:**
```bash
python Naive_SA.py
```

## Output Files

The system generates detailed analysis reports:

- **error_analysis.txt**: Misclassification analysis with actual vs predicted labels
- **LionKing_Output.txt**: Sentiment predictions with confidence scores for Lion King reviews

## Configuration Options

- **Stemming**: Enable/disable Porter stemming for word normalization
- **Train/Test Split**: Configurable ratio for model validation (default: 80/20)
- **Random Seed**: Reproducible results for consistent evaluation

## Technical Implementation

**Core Components:**
- `process_data()`: Text preprocessing and feature extraction
- `train_naive_bayes()`: Statistical model training with Laplace smoothing
- `predict_naive_bayes()`: Classification with confidence scoring
- `evaluate_classifier()`: Performance measurement and validation

**Statistical Methods:**
- Maximum likelihood estimation for probability calculation
- Log-space computation to prevent numerical underflow
- Add-1 smoothing for robust handling of unseen vocabulary
- Cross-validation through train/test dataset splitting

## Model Limitations

- **Independence Assumption**: Treats words as conditionally independent (naive assumption)
- **Linguistic Nuance**: May struggle with sarcasm, irony, and context-dependent meaning
- **Vocabulary Dependency**: Performance tied to training vocabulary coverage
- **Binary Classification**: Limited to positive/negative sentiment (no neutral category)

## Evaluation Results

The classifier demonstrates robust performance on movie review datasets with:
- Accurate classification of straightforward positive/negative sentiment
- Appropriate confidence scoring for prediction reliability
- Systematic error analysis for model improvement insights
- Statistical validation through comprehensive testing methodology

## License

This project is available under the MIT License.
