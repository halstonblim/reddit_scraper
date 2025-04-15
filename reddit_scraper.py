import os
import praw
import logging
from datetime import datetime
import pytz
import pandas as pd
from tqdm import tqdm

class RedditScraper:
    def __init__(self, config):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.timezone = pytz.timezone(config['timezone'])

    def get_posts(self, subreddit_config):
        """Fetch posts from a subreddit based on configuration."""
        subreddit_name = subreddit_config['name']
        limit = subreddit_config['post_limit']
        retrieved_at = datetime.now(self.timezone)
        records = []

        subreddit = self.reddit.subreddit(subreddit_name)
        
        logging.info(f"Fetching {limit} posts from r/{subreddit_name}")
        
        for submission in tqdm(subreddit.top(time_filter="day", limit=limit), 
                             total=limit, 
                             desc=f"Processing r/{subreddit_name}"):
            records.append({
                "subreddit": subreddit_name,
                "created_at": datetime.fromtimestamp(submission.created_utc, tz=self.timezone),
                "retrieved_at": retrieved_at,
                "type": "post",
                "text": submission.title + "\n\n" + submission.selftext,
                "score": submission.score,
                "post_id": submission.id,
            })

        return pd.DataFrame(records)

    def print_rate_limit_info(self):
        """Print current Reddit API rate limit information."""
        reset_ts = self.reddit.auth.limits.get('reset_timestamp')
        reset_time = datetime.fromtimestamp(reset_ts, tz=self.timezone).strftime("%Y-%m-%d %I:%M:%S %p %Z") if reset_ts else "Unknown"

        logging.info("Reddit API Rate Limit Info")
        logging.info(f"Requests used:      {self.reddit.auth.limits.get('used')}")
        logging.info(f"Requests remaining: {self.reddit.auth.limits.get('remaining')}")
        logging.info(f"Resets at:          {reset_time}") 