from dotenv import load_dotenv
import os
import requests
import functools
import json

load_dotenv()

api_key = os.getenv("NYT_API_KEY")


def send_api_call(funcGetURL):
    @functools.wraps(funcGetURL)
    # this just adds the documentation via .__doc__ but it makes me sound smart
    # so I'm keeping it
    def wrapper(*args, **kwargs):
        try:
            # wrap with some functionality
            response = requests.get(url=funcGetURL(*args, **kwargs))

            print(response, type(response), response.json(), type(response.json()))
            return response.json()
        except Exception as e:
            print(f"ERROR HAS OCCURED: {e}.")

    return wrapper


@send_api_call
def archive_api_endpoint(month, year) -> str:
    """
    Go through article archives. Not the most up-to-date but can maybe help with training data.

    @month (int/str): number from 1-12
    @year (int/str): 4-digit year
    https://developer.nytimes.com/docs/archive-product/1/overview
    """
    return (
        f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
    )


@send_api_call
def article_search_api_endpoint(query: str, search_filter=None) -> str:
    """
    Search for articles.

    @query (str): search terms
    @search_filter: honestly a bit confused here, will read up on documentation later but this seems super powerful.
    https://developer.nytimes.com/docs/articlesearch-product/1/overview
    """
    if search_filter:
        return f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&api-key={api_key}"

    return f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&fq={search_filter}&api-key={api_key}"


@send_api_call
def most_popular_api_endpoint(days):
    """
    Fetch most popular articles based on source (e.g. emailed, Facebook, etc.).

    Incomplete since there's so many endpoints. We can return the one(s) we need.

    @days (int/str): span of time where articles are sorted on popularity
    https://developer.nytimes.com/docs/most-popular-product/1/overview
    """

    return f"https://api.nytimes.com/svc/mostpopular/v2/emailed/{days}.json?api-key={api_key}"  # emailed


@send_api_call
def rss_feeds_endpoint(category: str = "Business") -> str:
    """
    RSS Feed that doesn't require an API key.

    @category (str): See documentation for allowed categories
    This is an xml request
    https://developer.nytimes.com/docs/rss-api/1/overview
    """
    return f"https://rss.nytimes.com/services/xml/rss/nyt/{category}.xml"


@send_api_call
def times_wire_endpoint(content_type: str, category: str = "Business") -> str:
    """
    Up to-the-minute stream of latest NYT articles.

    @content_type (str): all, nyt, inyt
    @category (str): category of news feed
    """
    return f"https://api.nytimes.com/svc/news/v3/content/{content_type}/{category}.json?api-key={api_key}"


@send_api_call
def top_stories_endpoint(category: str = "Business"):
    """
    Return top stories in selected category.

    @category (str): category of news feed
    """
    return (
        f"https://api.nytimes.com/svc/topstories/v2/{category}.json?api-key={api_key}"
    )


def remove_specific_key(json_dict, rubbish: set):
    # recursive deletion
    for item in rubbish:
        if item in json_dict:
            del json_dict[item]
    for key, value in json_dict.items():
        # check for rubbish in sub json_dict
        if isinstance(value, dict):
            remove_specific_key(value, rubbish)

        # check for existence of rubbish in lists
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_specific_key(item, rubbish)

    return json_dict


data = archive_api_endpoint("1", "2022")

# data = article_search_api_endpoint("US Stock Market")

remove_specific_key(data, rubbish=set(["multimedia", "keywords", "person"]))

with open("test1.json", "w") as f:
    json.dump(data, f)
