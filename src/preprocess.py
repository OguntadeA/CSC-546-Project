import re

# Clean review text before comparing reviews
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Split cleaned text into groups of consecutive words
def create_shingles(text, k=2):
    words = text.split()
    shingles = set()

    # Create each shingle and add it to our set
    for i in range(len(words) - k + 1):
        shingle = tuple(words[i:i + k])
        shingles.add(shingle)

    return shingles