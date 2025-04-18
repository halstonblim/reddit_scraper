import os
import praw
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi
import yaml
from pathlib import Path
from dotenv import load_dotenv

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with current date
    log_file = log_dir / f"reddit_scraper_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

class RedditScraper:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.timezone = pytz.timezone(config['timezone'])
        
        # Ensure output directory exists
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory set to: {config['output_dir']}")

    def get_posts(self, subreddit_config):
        """Fetch posts and comments from a subreddit based on configuration."""
        subreddit_name = subreddit_config['name']
        post_limit = subreddit_config['post_limit']
        comment_limit = subreddit_config['comment_limit']
        retrieved_at = datetime.now(self.timezone)
        records = []

        subreddit = self.reddit.subreddit(subreddit_name)
        
        self.logger.info(f"Fetching {post_limit} posts from r/{subreddit_name}")
        
        for submission in tqdm(subreddit.top(time_filter="day", limit=post_limit), 
                             total=post_limit, 
                             desc=f"Processing r/{subreddit_name}"):
            # Add post record
            records.append({
                "subreddit": subreddit_name,
                "created_at": datetime.fromtimestamp(submission.created_utc, tz=self.timezone),
                "retrieved_at": retrieved_at,
                "type": "post",
                "text": submission.title + "\n\n" + submission.selftext,
                "score": submission.score,
                "post_id": submission.id,
                "parent_id": None
            })

            # Get top comments if comment_limit > 0
            if comment_limit > 0:
                submission.comment_sort = 'top'
                submission.comments.replace_more(limit=0)  # Remove MoreComments objects
                for comment in submission.comments[:comment_limit]:
                    records.append({
                        "subreddit": subreddit_name,
                        "created_at": datetime.fromtimestamp(comment.created_utc, tz=self.timezone),
                        "retrieved_at": retrieved_at,
                        "type": "comment",
                        "text": comment.body,
                        "score": comment.score,
                        "post_id": comment.id,
                        "parent_id": comment.parent_id
                    })

        return pd.DataFrame(records)

    def print_rate_limit_info(self):
        """Print current Reddit API rate limit information."""
        reset_ts = self.reddit.auth.limits.get('reset_timestamp')
        reset_time = datetime.fromtimestamp(reset_ts, tz=self.timezone).strftime("%Y-%m-%d %I:%M:%S %p %Z") if reset_ts else "Unknown"

        self.logger.info("Reddit API Rate Limit Info")
        self.logger.info(f"Requests used:      {self.reddit.auth.limits.get('used')}")
        self.logger.info(f"Requests remaining: {self.reddit.auth.limits.get('remaining')}")
        self.logger.info(f"Resets at:          {reset_time}")

    def save_to_csv(self, df, date_str):
        """Save DataFrame to CSV file."""
        output_path = Path(self.config['output_dir']) / f"{date_str}.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")
        return output_path

    def save_to_parquet(self, df, date_str):
        """Save DataFrame to parquet file."""
        output_path = Path(self.config['output_dir']) / f"posts_{date_str}.parquet"
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")
        return output_path

    def upload_to_hf(self, df, date_str):
        if not self.config.get('push_to_hf', False):
            self.logger.info("Skipping Hugging Face upload as configured")
            return

        try:
            # Initialize HF API
            api = HfApi()
            repo_id = "hblim/top_reddit_posts_daily"
            
            # Try to get previous day's parquet file
            try:
                # Calculate previous day's date
                current_date = datetime.strptime(date_str, "%Y-%m-%d")
                prev_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
                prev_file_path = f"data_raw/{prev_date}.parquet"
                
                self.logger.info(f"Checking for previous day's file: {prev_file_path}")
                
                # Download previous day's parquet file
                downloaded_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename=prev_file_path,
                    repo_type="dataset",    # since you're fetching from a dataset repo
                )
                
                # Read post_ids from the previous day's parquet file
                existing_df = pd.read_parquet(downloaded_path)
                existing_ids = set(existing_df["post_id"].tolist())
                
                # Clean up temp file
                Path(downloaded_path).unlink()
                
                # Remove duplicates from current day's data
                original_count = len(df)
                df = df[~df["post_id"].isin(existing_ids)]
                filtered_count = len(df)
                
                self.logger.info(f"Filtered {original_count - filtered_count} duplicate posts based on post_id")
                
                if df.empty:
                    self.logger.info("No new posts to upload after deduplication")
                    return
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch or process previous day's file: {str(e)}")
                # Continue with current day's data if previous day's file doesn't exist or can't be processed
            
            # Save to parquet locally
            parquet_path = self.save_to_parquet(df, date_str)
            
            # Prepare path in repo
            path_in_repo = f"data_raw/{date_str}.parquet"
            
            # Upload to HF
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN")
            )
            self.logger.info(f"Uploaded {len(df)} rows for {date_str} → {path_in_repo}")

        except Exception as e:
            self.logger.error(f"Failed to upload to Hugging Face: {str(e)}", exc_info=True)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Reddit scraper...")

    # Check for required environment variables
    required_env_vars = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT"
    ]
    
    # Only require HF_TOKEN if push_to_hf is True
    config = load_config()
    if config.get('push_to_hf', False):
        required_env_vars.append("HF_TOKEN")
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Initialize scraper
    logger.info("Initializing Reddit scraper...")
    scraper = RedditScraper(config)
    
    # Get current date in YYYY-MM-DD format
    date_str = datetime.now(pytz.timezone(config['timezone'])).strftime("%Y-%m-%d")
    logger.info(f"Processing data for date: {date_str}")
    
    # Collect all posts and comments
    all_records = []
    for subreddit_config in config['subreddits']:
        logger.info(f"Processing subreddit: {subreddit_config['name']}")
        records_df = scraper.get_posts(subreddit_config)
        all_records.append(records_df)
    
    # Combine all records
    combined_df = pd.concat(all_records, ignore_index=True)
    logger.info(f"Total records collected: {len(combined_df)}")
    
    # Save to CSV
    scraper.save_to_csv(combined_df, date_str)
    
    # Upload to Hugging Face if configured
    scraper.upload_to_hf(combined_df, date_str)
    
    # Print rate limit info
    scraper.print_rate_limit_info()
    
    logger.info("Reddit scraper completed successfully")

if __name__ == "__main__":
    main() 