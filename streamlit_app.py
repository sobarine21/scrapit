import os
import time
import base64
import pandas as pd
import streamlit as st
from queue import Queue
from threading import Thread, Lock
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from google import genai
from google.genai import types

# ---- CONFIG ----

MAX_CONCURRENT_REQUESTS = 15
API_KEYS = st.secrets["GEMINI"]["GEMINI_API_KEYS"]
lock = Lock()

# ---- Gemini API Content Generation ----

def generate_ai_insights(prompt, api_key):
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-flash-lite"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
        ],
    )

    output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        output += chunk.text

    return output.strip()

# ---- Selenium Setup ----

def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# ---- Worker Thread Function ----

def process_url(task_queue, result_list, knowledge_text, key_rotation):
    while not task_queue.empty():
        url = task_queue.get()
        if not url:
            task_queue.task_done()
            continue

        driver = None  # Initialize driver to None

        try:
            driver = setup_selenium()
            driver.get(url)
            time.sleep(3)  # Allow full page load

            page_source = driver.page_source

            prompt = f"""Based on the knowledge base provided, extract relevant regulatory enforcement or actions data from the webpage content below.

Knowledge Base:
{knowledge_text}

Page Content:
{page_source}
"""

            # Rotate API key safely
            with lock:
                api_key = key_rotation.pop(0)
                key_rotation.append(api_key)

            ai_response = generate_ai_insights(prompt, api_key)

            result_list.append({
                "url": url,
                "ai_extracted_data": ai_response
            })

        except Exception as e:
            result_list.append({
                "url": url,
                "ai_extracted_data": f"ERROR: {str(e)}"
            })

        finally:
            if driver:
                driver.quit()
            task_queue.task_done()

# ---- Streamlit App Interface ----

def main():
    st.title("AI-Powered Regulatory Data Scraper")

    uploaded_csv = st.file_uploader("Upload CSV (must contain column 'url')", type=["csv"])
    knowledge_file = st.file_uploader("Upload Knowledge Base Text File (.txt)", type=["txt"])

    if uploaded_csv and knowledge_file:
        urls_df = pd.read_csv(uploaded_csv)
        knowledge_text = knowledge_file.read().decode("utf-8")

        if "url" not in urls_df.columns:
            st.error("CSV must contain a column named 'url'.")
            return

        urls = urls_df["url"].dropna().tolist()
        st.info(f"Total URLs to process: {len(urls)}")

        if st.button("Start Processing"):

            result_list = []
            task_queue = Queue()

            for url in urls:
                task_queue.put(url)

            key_rotation = API_KEYS.copy()

            threads = []
            num_threads = min(MAX_CONCURRENT_REQUESTS, len(urls))

            for _ in range(num_threads):
                t = Thread(target=process_url, args=(task_queue, result_list, knowledge_text, key_rotation))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            result_df = pd.DataFrame(result_list)
            st.dataframe(result_df)

            csv_data = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()

            st.markdown(
                f'<a href="data:file/csv;base64,{b64}" download="extracted_data.csv">⬇️ Download Results as CSV</a>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
