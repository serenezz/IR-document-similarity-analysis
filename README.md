# Document Similarity Analysis using Vector Space Model and Multithreading Computation

### Overview
This project is an implementation of a document similarity analysis using the Vector Space Model (VSM). It focuses on text preprocessing, term frequency (TF) calculation, and computing document similarity by using multiple weighting techniques through multithreading: TF, TF-IDF, and WF-IDF.

### Features
- **Implementing and using the Vector Space Model**
- **Implementing similarity between documents**
- Text tokenization, stopword removal, and stemming
- Term frequency (TF) matrix calculation
- Cosine similarity computation using different weighting schemes (TF, TF-IDF, WF-IDF)
- Multithreading for parallel computation of similarity measures

### Multithreading Implementation
To optimize performance, multithreading is employed to compute document similarities in parallel. Three separate threads are used:
1. **Thread 1:** Computes document similarity using TF.
2. **Thread 2:** Computes document similarity using TF-IDF.
3. **Thread 3:** Computes document similarity using WF-IDF.

By leveraging the `ThreadPoolExecutor`, the computation of similarity measures is efficiently parallelized, significantly reducing execution time when working with large document sets.

### Usage Instructions
1. Open the terminal in the folder.
2. Run the script using:
   ```bash
   python 3.py 10
   ```
   Here, `10` represents the number of top frequent terms or document similarities to display. You can replace it with any desired number.

### Requirements
- Python 3.x
- `nltk`
- `pandas`
- `concurrent.futures`

### Output
The script will display:
- The most frequent terms in the document set.
- The top `k` most similar document pairs using different similarity measures.
