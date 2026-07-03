from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import praw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Galaxy S25 OR Samsung S25")
    parser.add_argument("--subreddit", default="Android")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/reddit_comments_unlabeled.csv"))
    args = parser.parse_args()

    reddit_config = {
        "client_id": os.environ["REDDIT_CLIENT_ID"],
        "client_secret": os.environ["REDDIT_API_SECRET"],
        "user_agent": os.environ.get("REDDIT_USER_AGENT", "android-galaxy-sentiment-script"),
    }
    reddit = praw.Reddit(**reddit_config)

    rows = []
    for submission in reddit.subreddit(args.subreddit).search(args.query, limit=args.limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            rows.append({
                "submission_id": submission.id,
                "submission_title": submission.title,
                "comment_text": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
                "subreddit": str(submission.subreddit),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
