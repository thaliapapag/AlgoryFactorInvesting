# AlgoryFactorInvesting

Factor investing model, testing 100 alphas and BERT sentiment analysis.

Install required Python packages.

```
pip install -r requirements.txt
```

Run data_download.py to get ticker data.

## Approach

- Calculate alphas and load into dataframes with aligned datetime index

## NYT API

[NYT API Documentation](https://developer.nytimes.com/apis)

**Currently enabled:**

- Archive API
- Article Search API
- Most Popular API
- RSS Feeds
- Times Wire API

### To-Do

Get alphas from PDF

- Alex
  - Get WSJ API working. Begin preprocessing data for sentiment analysis
  - 76-100 alphas
- Thalia
  - Bloomberg access
  - 51-75 alphas
- Wendy
  - 0-25 alphas
- Peter
  - 26-50 alphas

<!--
### Reset .gitignore if it doesn't work

```
git rm -rf --cached .
git add .
```
-->
