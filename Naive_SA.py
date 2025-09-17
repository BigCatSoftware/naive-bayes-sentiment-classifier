"""
Naive Bayes Sentiment Classification System

Statistical text classification using probabilistic modeling for sentiment analysis.
Implements Laplace smoothing and comprehensive evaluation metrics.
"""

import os
import math
import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def download_nltk_resources():
    """Download necessary NLTK resources for text processing."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"Warning: Could not download NLTK resources: {e}")


def load_stopwords():
    """Load English stopwords using NLTK."""
    try:
        return set(stopwords.words('english'))
    except Exception as e:
        print(f"Error loading NLTK stopwords: {e}")
        print("Please ensure NLTK is installed and stopwords are downloaded.")
        raise


def process_data(review, use_stemming=False):
    """
    Process a single review using NLTK tokenization and stopword removal.

    Args:
        review (str): Raw text review to process
        use_stemming (bool): Whether to apply Porter stemming

    Returns:
        list: Cleaned and processed words
    """
    stopwords_set = load_stopwords()

    # Convert to lowercase
    review = review.lower()

    # Use NLTK tokenization
    tokens = word_tokenize(review)

    # Remove punctuation and stopwords
    words = []
    stemmer = PorterStemmer() if use_stemming else None

    for token in tokens:
        # Keep only alphabetic tokens that aren't stopwords
        if token.isalpha() and token not in stopwords_set:
            if use_stemming and stemmer:
                token = stemmer.stem(token)
            words.append(token)

    return words


def load_reviews_from_folder(folder_path, label):
    """
    Load all review files from a folder and return labeled dataset.

    Args:
        folder_path (str): Path to folder containing review text files
        label (int): Class label for reviews (0 for negative, 1 for positive)

    Returns:
        list: List of (review_text, label) tuples
    """
    reviews = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        return reviews

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    reviews.append((content, label))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return reviews


def count_reviews(reviews, use_stemming=False):
    """
    Count word frequencies for each class to build feature vectors.

    Args:
        reviews (list): List of (review_text, label) tuples
        use_stemming (bool): Whether to apply stemming during processing

    Returns:
        dict: Dictionary with (word, class) as key and count as value
    """
    word_counts = defaultdict(int)

    for review, label in reviews:
        words = process_data(review, use_stemming)
        for word in words:
            word_counts[(word, label)] += 1

    return dict(word_counts)


def train_naive_bayes(word_counts, reviews):
    """
    Train Naive Bayes classifier using maximum likelihood estimation with Laplace smoothing.

    Mathematical formulation:
    - Prior probabilities: P(class) = count(class) / total_documents
    - Likelihood: P(word|class) = (count(word,class) + 1) / (total_words_in_class + vocabulary_size)
    - Log-likelihood ratio: log(P(word|pos)) - log(P(word|neg))

    Args:
        word_counts (dict): Word frequency counts per class
        reviews (list): Training dataset

    Returns:
        tuple: (logprior, loglikelihood_dict) for classification
    """
    # Calculate class distributions
    total_docs = len(reviews)
    pos_docs = sum(1 for _, label in reviews if label == 1)
    neg_docs = sum(1 for _, label in reviews if label == 0)

    # Compute log prior probabilities
    logprior = math.log(pos_docs / total_docs) - math.log(neg_docs / total_docs)

    # Calculate total words per class for normalization
    pos_words = sum(count for (word, label), count in word_counts.items() if label == 1)
    neg_words = sum(count for (word, label), count in word_counts.items() if label == 0)

    # Extract vocabulary
    vocabulary = set(word for word, label in word_counts.keys())
    vocab_size = len(vocabulary)

    print(f"Training statistics:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Total positive words: {pos_words}")
    print(f"  Total negative words: {neg_words}")
    print(f"  Class balance: {pos_docs}/{neg_docs} (pos/neg)")

    # Compute log-likelihood ratios with Laplace smoothing
    loglikelihood = {}
    for word in vocabulary:
        freq_pos = word_counts.get((word, 1), 0)
        freq_neg = word_counts.get((word, 0), 0)

        # Apply Laplace smoothing (add-1)
        prob_pos = (freq_pos + 1) / (pos_words + vocab_size)
        prob_neg = (freq_neg + 1) / (neg_words + vocab_size)

        # Store log-likelihood ratio
        loglikelihood[word] = math.log(prob_pos) - math.log(prob_neg)

    return logprior, loglikelihood


def predict_naive_bayes(review, logprior, loglikelihood, use_stemming=False):
    """
    Classify sentiment of a review using trained Naive Bayes model.

    Decision rule: score = logprior + sum(loglikelihood[word] for word in review)
    Classification: positive if score > 0, negative otherwise

    Args:
        review (str): Text to classify
        logprior (float): Log prior probability ratio
        loglikelihood (dict): Log-likelihood ratios for vocabulary
        use_stemming (bool): Whether to apply stemming

    Returns:
        float: Classification score (positive = positive sentiment)
    """
    words = process_data(review, use_stemming=use_stemming)

    # Sum log-likelihoods of words present in vocabulary
    log_sum = logprior
    for word in words:
        if word in loglikelihood:
            log_sum += loglikelihood[word]

    return log_sum


def evaluate_classifier(test_reviews, logprior, loglikelihood, use_stemming=False):
    """
    Evaluate classifier performance on test dataset.

    Args:
        test_reviews (list): Test dataset as (review, label) pairs
        logprior (float): Trained log prior
        loglikelihood (dict): Trained log-likelihood ratios
        use_stemming (bool): Whether to apply stemming

    Returns:
        tuple: (accuracy, predictions_list)
    """
    correct = 0
    total = len(test_reviews)
    predictions = []

    for review, actual_label in test_reviews:
        prediction_score = predict_naive_bayes(review, logprior, loglikelihood, use_stemming)
        predicted_label = 1 if prediction_score > 0 else 0
        predictions.append((review, actual_label, predicted_label, prediction_score))

        if predicted_label == actual_label:
            correct += 1

    accuracy = correct / total
    return accuracy, predictions


def analyze_classification_errors(predictions, output_file='error_analysis.txt'):
    """
    Perform error analysis and generate misclassification report.

    Args:
        predictions (list): List of prediction tuples
        output_file (str): Output file for error report
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Classification Error Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Actual':<8} {'Predicted':<10} {'Score':<8} Review\n")
        f.write("-" * 80 + "\n")

        error_count = 0
        for review, actual_label, predicted_label, score in predictions:
            if predicted_label != actual_label:
                error_count += 1
                # Clean and truncate review for readability
                review_text = review.replace('\n', ' ').replace('\t', ' ')
                if len(review_text) > 80:
                    review_text = review_text[:80] + "..."

                actual_str = "Positive" if actual_label == 1 else "Negative"
                pred_str = "Positive" if predicted_label == 1 else "Negative"

                f.write(f"{actual_str:<8} {pred_str:<10} {score:<8.3f} {review_text}\n")

        f.write(f"\nError Summary:\n")
        f.write(f"Total misclassifications: {error_count}\n")
        f.write(f"Error rate: {error_count/len(predictions):.3f}\n")

    print(f"Error analysis complete. {error_count} misclassifications saved to {output_file}")


def predict_sentiment_from_file(filename, logprior, loglikelihood,
                                output_file='sentiment_predictions.txt', use_stemming=False):
    """
    Apply trained model to classify sentiment of reviews from file.

    Args:
        filename (str): Input file containing reviews (one per line)
        logprior (float): Trained log prior
        loglikelihood (dict): Trained log-likelihood ratios
        output_file (str): Output file for predictions
        use_stemming (bool): Whether to apply stemming
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f.readlines() if line.strip()]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Sentiment Analysis Results\n")
            f.write("=" * 50 + "\n\n")

            positive_count = 0
            negative_count = 0

            for i, review in enumerate(reviews, 1):
                if review:
                    prediction_score = predict_naive_bayes(review, logprior, loglikelihood, use_stemming)
                    sentiment = "Positive" if prediction_score > 0 else "Negative"
                    confidence = abs(prediction_score)

                    if prediction_score > 0:
                        positive_count += 1
                    else:
                        negative_count += 1

                    f.write(f"Review {i}:\n")
                    f.write(f'"{review}"\n')
                    f.write(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})\n")
                    f.write("-" * 50 + "\n")

                    print(f"Review {i}: {sentiment} (Confidence: {confidence:.4f})")

            # Summary statistics
            f.write(f"\nSummary:\n")
            f.write(f"Total reviews analyzed: {len(reviews)}\n")
            f.write(f"Positive sentiment: {positive_count}\n")
            f.write(f"Negative sentiment: {negative_count}\n")
            f.write(f"Positive ratio: {positive_count/len(reviews):.3f}\n")

    except FileNotFoundError:
        print(f"Error: {filename} not found.")


def main():
    """
    Main execution pipeline for sentiment classification system.
    Performs data loading, model training, evaluation, and prediction.
    """
    print("Naive Bayes Sentiment Classification System")
    print("=" * 50)

    # Initialize NLTK resources
    download_nltk_resources()

    # Data loading
    print("\n=== Data Loading ===")
    pos_reviews = load_reviews_from_folder('pos/pos', 1)
    neg_reviews = load_reviews_from_folder('neg/neg', 0)

    # Combine and validate dataset
    all_reviews = pos_reviews + neg_reviews

    if not all_reviews:
        print("No reviews found. Please check your folder structure.")
        print("Expected structure:")
        print("  pos/pos/  (containing positive review .txt files)")
        print("  neg/neg/  (containing negative review .txt files)")
        return

    print(f"Dataset loaded: {len(pos_reviews)} positive, {len(neg_reviews)} negative reviews")

    # Train/test split with reproducible randomization
    random.seed(42)
    random.shuffle(all_reviews)
    split_point = int(0.8 * len(all_reviews))
    train_reviews = all_reviews[:split_point]
    test_reviews = all_reviews[split_point:]

    print(f"Training set: {len(train_reviews)} reviews")
    print(f"Test set: {len(test_reviews)} reviews")

    # Feature extraction
    print("\n=== Feature Extraction ===")
    use_stemming = False  # Configure stemming option
    word_counts = count_reviews(train_reviews, use_stemming)
    print(f"Extracted {len(word_counts)} word-class feature pairs")

    # Model training
    print("\n=== Model Training ===")
    logprior, loglikelihood = train_naive_bayes(word_counts, train_reviews)
    print(f"Model trained with log-prior: {logprior:.4f}")
    print(f"Vocabulary size: {len(loglikelihood)}")

    # Model evaluation
    print("\n=== Model Evaluation ===")
    accuracy, predictions = evaluate_classifier(test_reviews, logprior, loglikelihood, use_stemming)
    print(f"Classification accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Validate model with sample input
    sample_review = "This movie was absolutely fantastic!"
    sample_score = predict_naive_bayes(sample_review, logprior, loglikelihood, use_stemming)
    sample_sentiment = "Positive" if sample_score > 0 else "Negative"
    print(f"Sample prediction: '{sample_review}' → {sample_sentiment} (score: {sample_score:.4f})")

    # Performance analysis
    print("\n=== Performance Analysis ===")
    analyze_classification_errors(predictions)

    # Apply model to external data
    print("\n=== Sentiment Prediction ===")
    external_reviews_file = 'LionKing_MovieReviews.txt'
    if os.path.exists(external_reviews_file):
        predict_sentiment_from_file(external_reviews_file, logprior, loglikelihood,
                                   output_file='LionKing_Sentiment_Analysis.txt', use_stemming=use_stemming)
        print(f"Lion King predictions saved to LionKing_Sentiment_Analysis.txt")
    else:
        print(f"External review file '{external_reviews_file}' not found. Skipping prediction phase.")

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("  - error_analysis.txt")
    print("  - LionKing_Sentiment_Analysis.txt")


if __name__ == "__main__":
    main()