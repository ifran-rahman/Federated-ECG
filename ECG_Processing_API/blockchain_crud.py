import requests

# Replace 'YOUR_API_ENDPOINT' with the actual URL of the API endpoint you want to post to
api_endpoint = 'http://localhost:3210/asset'


def save_data(data):
    try:
        response = requests.post(api_endpoint, json=data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Data was successfully posted to the API.")
        else:
            print(f"Failed to post data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


