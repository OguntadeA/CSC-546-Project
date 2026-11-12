def jaccard_similarity(set_a, set_b):
    #Calculate exact Jaccard similarity between two sets.
    intersection = set_a & set_b
    #Find all unique items that appear in EITHER set.
    union = set_a | set_b
    if not union:
        return 1.0
    # Divide the number of shared items by the total
    # number of unique items to calculate similarity.
    return len(intersection) / len(union)