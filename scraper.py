from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

service = Service(ChromeDriverManager().install())
options = Options()
#options.add_argument("--headless")  # Run headless to avoid opening a browser window
driver = webdriver.Chrome(service=service, options=options)

# used url = "https://www.reddit.com/search/?q=larsen+and+toubro&type=link" as well
url = "https://www.reddit.com/search/?q=l%26t&type=link"
driver.get(url)
time.sleep(5)

def reached_bottom():
    return driver.execute_script(
        "(window.innerHeight + window.scrollY) >= document.body.offsetHeight"
    )
initial_y = driver.execute_script("return window.scrollY")
while True:
    current_y = driver.execute_script("return window.scrollY")
    driver.execute_script("window.scrollBy(0, 500);")
    time.sleep(1)
    if current_y == driver.execute_script("return window.scrollY"):
        print("Stopped scrolling.")
        break
    if reached_bottom():
        print("Reached the bottom of the page.")
        break

time.sleep(2)
script = """
    var ariaLabels = [];
    var hrefs = [];
    var postTitles = document.querySelectorAll('[data-testid="post-title"]');
    console.log("Number of post titles found:", postTitles.length);
    for (var i = 0; i < postTitles.length; i++) {
        var postTitle = postTitles[i];
        var ariaLabel = postTitle.getAttribute('aria-label');
        var href = postTitle.getAttribute('href');
        if (ariaLabel && href) {
            ariaLabels.push(ariaLabel);
            hrefs.push(href);
        }
    }
    return [ariaLabels, hrefs];
"""
post_titles_text, post_links = driver.execute_script(script)

posts_text = []
posts_upvotes = []
for i in range(len(post_titles_text)):
    service = Service(ChromeDriverManager().install())
    options = Options()
    #options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=options)
    url = post_links[i]
    driver.get("https://www.reddit.com"+url)
    post_id = url.split('/')[4]

    extract_text = f'''
    var element = document.querySelector("#t3_{post_id}-post-rtjson-content");
    return element ? element.textContent : "Text not found";
    '''
    post_text = driver.execute_script(extract_text)

    shadow_root_script = f"""
    var host = document.querySelector("#t3_{post_id}");
    var shadow = host.shadowRoot;
    return shadow.querySelector("faceplate-number").innerText;
    """
    post_upvotes = driver.execute_script(shadow_root_script)
    posts_text.append(post_text)
    posts_upvotes.append(post_upvotes)
    driver.quit()

rows = zip(post_titles_text, post_links, post_text, post_upvotes)
filename = "reddit_posts.csv"
data = zip(post_titles_text, post_links, posts_text, posts_upvotes)
csv_file_path = "reddit_posts_data.csv"

with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Post Title', 'Post Link', 'Post Text', 'Post Upvotes'])
    for title, link, text, upvotes in data:
        if text:
            writer.writerow([title, link, text, upvotes])
        else:
            writer.writerow([title, link, "No post text available", upvotes])

print("Data has been written to", csv_file_path)
print(f"Data has been written to {filename} successfully!")