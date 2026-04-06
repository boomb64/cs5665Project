import requests

# Test fetching the JSON file for the P361 bucket
url = "https://aicuneiform.com/p/p361.json"
data = requests.get(url).json()

# Let's print out the data specifically for P361099 to see how they labeled the English text
if "P361099" in data:
    print(data["P361099"])
elif type(data) == list:
    print(data[0])
else:
    print(data.keys())