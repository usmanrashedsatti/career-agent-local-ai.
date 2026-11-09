\# Career Agent: Private AI Career Coach



A personal AI assistant for job seekers in data science. It runs \*\*fully on your own computer\*\* using a local LLM (Ollama), so your CV and notes never leave your machine.



\## Why I built it

I am a Data Scientist (MSc Data Science with Advanced Research, University of Hertfordshire) looking for roles in Saudi Arabia. Searching for jobs meant tailoring a CV for every ad, tracking applications and practising interviews. This project puts those three tasks in one private tool.



\## Features

| Tab | What it does |

|---|---|

| \*\*Dashboard\*\* | Application stats, daily goal and quick actions |

| \*\*Ask Me\*\* | Answers questions using only your own files (RAG) |

| \*\*Job Tracker\*\* | Records applications and flags follow-ups after 7 days |

| \*\*CV Tailor\*\* | Paste a job ad, get a tailored summary, skill gaps, cover letter and recruiter message |

| \*\*Interview Coach\*\* | Mock interview, SQL practice, statistics and ML quiz, STAR practice, with scoring and saved sessions |



\## How it works

1\. Documents (PDF, TXT, MD) in `data/` are split into chunks.

2\. Each chunk is turned into an embedding with `nomic-embed-text`.

3\. For each question, the most similar chunks are retrieved by cosine similarity.

4\. A local model (`llama3.2:3b`) answers using only those chunks. The prompts tell it not to invent facts.



Memory is stored in a simple `memory.json` file, so it runs on 32-bit Python without a vector database.



\## Tech stack

Python, Ollama, llama3.2:3b, nomic-embed-text, pypdf, customtkinter, JSON storage.



\## Setup (Windows)

1\. Install \[Ollama](https://ollama.com) and Python 3.10 or newer.

2\. Download the models:

&#x20;  ```

&#x20;  ollama pull llama3.2:3b

&#x20;  ollama pull nomic-embed-text

&#x20;  ```

3\. Install the libraries:

&#x20;  ```

&#x20;  pip install -r requirements.txt

&#x20;  ```

4\. Put your CV and notes (text-based PDF, TXT or MD) in the `data/` folder.

5\. Run the desktop app:

&#x20;  ```

&#x20;  python app\_pro.py

&#x20;  ```



`agent.py` is a minimal terminal version of the same idea.



\## Privacy

\- Everything runs locally. No data is sent to a cloud service.

\- Do not put passports, ID numbers or bank details in `data/`.

\- Personal files are excluded by `.gitignore`.



\## Limitations

\- A 3B-parameter local model makes mistakes. Always check generated CV text before sending it.

\- It does not apply to jobs or control LinkedIn. It prepares material for you to submit.

\- Scanned PDFs are not supported (no OCR).

\- Reminders show only while the app is open.



\## Roadmap

\- RFQ document extraction tool

\- OCR for scanned PDFs

\- Larger local model option

\- Export tailored CVs to Word and PDF



\## Author

Muhammad Usman Rashed Satti, Data Scientist.

\[LinkedIn](https://www.linkedin.com/in/muhammad-usman-satti) | \[GitHub](https://github.com/usmanrashedsatti)

