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

def generate_ai_code(prompt, api_key):
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

        driver = None

        try:
            # Step 1: Visit URL and extract page source
            driver = setup_selenium()
            driver.get(url)
            time.sleep(3)
            page_source = driver.page_source

            # Step 2: Generate Python scraping script dynamically
            prompt = f"""Based on the following knowledge base and webpage HTML content, generate a Python function called 'scrape_page(html_content)' that extracts the relevant regulatory enforcement or actions data in structured dict format.

Knowledge Base:
{knowledge_text}

Page HTML Content:
{page_source}

Provide only the Python function code without explanations.
Example expected output: {{'enforcement_title': '...', 'date': '...', 'details': '...'}}
"""

            with lock:
                api_key = key_rotation.pop(0)
                key_rotation.append(api_key)

            generated_code = generate_ai_code(prompt, api_key)

            # Step 3: Safely execute the generated code and call scrape_page(html_content)
            local_vars = {}
            exec(generated_code, {}, local_vars)  # Run code in isolated namespace

            if "scrape_page" not in local_vars:
                raise Exception("Generated code did not provide 'scrape_page' function.")

            extracted_data = local_vars["scrape_page"](page_source)

            result_list.append({
                "url": url,
                "ai_generated_code": generated_code,
                "extracted_data": extracted_data
            })

        except Exception as e:
            result_list.append({
                "url": url,
                "ai_generated_code": generated_code if 'generated_code' in locals() else "N/A",
                "extracted_data": f"ERROR: {str(e)}"
            })

        finally:
            if driver:
                driver.quit()
            task_queue.task_done()

# ---- Streamlit Interface ----

def main():
    st.title("AI-Powered Regulatory Data Scraper with Dynamic Script Generation")

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
