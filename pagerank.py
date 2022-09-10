import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create dictionary to return
    # Get the length of links for current given page
    d = {}
    links = len(corpus[page])

    # If we have at least one or more links on this page
    # then do the following, otherwise go to 'else'
    if links:
        prob = damping_factor / links
        additional_prob = (1 - damping_factor) / len(corpus)

        # All pages get additional probablity added to them
        for page_link in corpus:
            d[page_link] = additional_prob

        # Each page connected by the page get prob added to it
        for page_link in corpus[page]:
            d[page_link] += prob

    # Every page is equally chosen
    else:
        prob = 1 / len(corpus)

        # Add the same probabilty to each page
        for p in corpus:
            d[p] = prob

    return d


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Set up variables dictionary, list, and random page chosen
    d = {}
    all_pages = list(corpus)
    random_page = random.choice(all_pages)

    # Set all pages to 0
    for page in corpus:
        d[page] = 0

    # Repeat this process n times for more accurate result
    for i in range(n):
        # If the page was chosen add 1 to it
        d[random_page] += 1

        # Get model results for the random page chosen
        model = transition_model(corpus, random_page, damping_factor)

        # Get a random page from the weighted model 'ranks'
        random_page = random.choices(all_pages, model.values())[0]

    # Divide all page values by N to normalize values
    for page in d:
        d[page] /= n

    return d


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Set up variables
    converged = False
    d = {}
    # summation, converted_count = 0, 0

    # Set up constants
    N = len(corpus)
    random_odds = 1 / N
    odds = (1-damping_factor) / N

    # Every page gets the same starting value
    for page in corpus:
        d[page] = random_odds

    # Keep iterating while we haven't converged
    while not converged:
        # Reset the converted count
        converted_count = 0

        # Get the page rank for every page
        for page in corpus:
            # Reset the summation
            summation = 0

            # Go check all the pages and see if it has a link to 'page'
            for link in corpus:
                # If it has a link to 'page' add it to the summation
                if page in corpus[link]:
                    # add to the summation using the provided formula
                    summation += damping_factor * d[link] / len(corpus[link])

            # Check to see if 'page' has converged, if so add 1 to count
            if abs(d[page] - (odds + summation)) < 0.001:
                converted_count += 1

            # Update the page value with new iteration page value
            d[page] = odds + summation

        # If all pages have converged on this loop, then update 'converged' to True
        if converted_count == N:
            converged = True

    return d


if __name__ == "__main__":
    main()
