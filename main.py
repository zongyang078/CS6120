# Download the data
def download_data():
    import ssl, urllib.request
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://course.ccs.neu.edu/cs6120s26/data/shakespeare/shakespeare-edit.txt"
    urllib.request.urlretrieve(url, "shakespeare-edit.txt")