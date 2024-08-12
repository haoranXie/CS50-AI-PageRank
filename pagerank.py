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
    totalPages = len(corpus)
    numLinks = len(corpus[page])

    probDistribution = dict()
    for page in corpus:
        probDistribution[page] = (1-damping_factor) / totalPages
        if page in numLinks:
            probDistribution[page] += damping_factor / numLinks
    return probDistribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = {page: 0 for page in corpus}
    currentPage = random.choice(list(corpus.keys()))
    for sample in range(n):
        pageRanks[currentPage] +=1
        model = transition_model(corpus, currentPage, damping_factor)
        currentPage = random.choices(pages = list(model.keys()),weights = list(model.values()), k = 1)[0]
    pageRanks = {page: rank/n for page, rank in pageRanks.items()}




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    totalPages = len(corpus)
    pageRanks = {page: 1/totalPages for page in corpus}    
    change = float('inf')

    while change>0.001:
        newRanks = {}
        for page in corpus:
            pageProbability = (1-damping_factor) / totalPages
            for possiblePage in corpus:
                if page in corpus[possiblePage]: #If this possible page is linked to the current page
                    pageProbability += damping_factor * pageRanks[possiblePage] / len(corpus[possiblePage]) #Add probability of going to the current page
                elif not corpus[possiblePage]: #If this possible page has no outbound links
                    pageProbability += damping_factor * pageRanks[possiblePage] / totalPages #Add probability of jumping randomly to the current page
            newRanks[page] = pageProbability
        change = max(abs(newRanks[page] - pageRanks[page]) for page in pageRanks)
        pageRanks = newRanks
    return pageRanks


if __name__ == "__main__":
    main()
