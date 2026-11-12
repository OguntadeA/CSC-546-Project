from datasketch import MinHash

# Create a MinHash signature for a review
# This helps us estimate similarity without comparing full shingle sets
def create_minhash(shingles, num_perm=128):
    minhash = MinHash(num_perm=num_perm)

    # Add each shingle to the MinHash signature
    for shingle in sorted(shingles):
        shingle_text = " ".join(shingle)
        minhash.update(shingle_text.encode("utf-8"))

    return minhash