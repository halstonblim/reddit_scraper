# reddit-scraper

A daily scraper that collects the top posts and comments from configurable subreddits, deduplicates against the previous day’s data, and optionally pushes each day’s results to a Hugging Face Hub dataset.

---

## Features

- Fetch top *N* posts and top *M* comments per subreddit via PRAW  
- Stamp each record with creation time and retrieval time (in your local timezone)  
- De‑duplicate today’s posts against yesterday’s shard stored on Hugging Face  
- Save each day’s data as both CSV and Parquet  
- Optional: upload the daily Parquet file to your HF dataset repo  
- Automatic logging with per‑day log files  

---

## Requirements

- Python 3.8+  
- [`praw`](https://praw.readthedocs.io/)  
- `pandas`, `pyarrow`  
- `tqdm`  
- `pytz`  
- [`datasets`](https://github.com/huggingface/datasets)  
- [`huggingface_hub`](https://github.com/huggingface/huggingface_hub)  
- `pyyaml`  
- `python-dotenv`  

Install with:

```bash
pip install praw pandas pyarrow tqdm pytz datasets huggingface_hub pyyaml python-dotenv
```

---

## Configuration

1. **Environment variables** (via a `.env` file in project root):

   ```dotenv
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=your_app_name_or_contact
   HF_TOKEN=your_huggingface_token      # only if push_to_hf: true
   ```

2. **`config.yaml`** (example):

   ```yaml
   timezone: "US/Central"
   output_dir: "data_raw"
   push_to_hf: true
   subreddits:
     - name: "googlepixel"
       post_limit: 50
       comment_limit: 5
     - name: "apple"
       post_limit: 30
       comment_limit: 3
   ```

---

## Usage

Run the scraper script directly:

```bash
python scrape.py
```

What happens:

1. Loads your `.env` and `config.yaml`  
2. Initializes logging (`logs/reddit_scraper_YYYY-MM-DD.log`)  
3. Fetches posts & comments for each subreddit in your config  
4. Saves combined results as:
   - `data_raw/YYYY-MM-DD.csv`
   - `data_raw/posts_YYYY-MM-DD.parquet`
5. If `push_to_hf: true`, then:
   - Downloads yesterday’s `data_raw/YYYY-1.parquet` from your HF repo  
   - Drops any duplicate `post_id`s  
   - Uploads today’s filtered Parquet to HF under `data_raw/YYYY-MM-DD.parquet`  
6. Prints your Reddit API rate‑limit info  

---

## Usage

```bash
# in config.yaml, set push_to_hf: false for no HF upload
python scrape.py
```

---

## Logs

All INFO, WARNING, and ERROR messages are written to `logs/reddit_scraper_YYYY-MM-DD.log`. Use these logs to diagnose failures or API‑limit issues.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to copy, modify, or extend.  