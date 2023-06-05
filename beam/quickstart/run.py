def hello_world(**inputs):
    print("You'll see these logs in the web dashboard 👀 This is some new logs by helin")
    return {"response": inputs["text"]}


if __name__ == "__main__":
    text = "Testing 123"
    hello_world(text=text)
