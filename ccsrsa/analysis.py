
import praw
import nltk
import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from constants import *

vader_analyzer = SentimentIntensityAnalyzer()

def main():

  reddit_config = json.load(open(PRAW_CONFIG_FILE))
  reddit = praw.Reddit(**reddit_config)

  for ccsr in CRYPTOCURRENCY_SUBREDDITS:
    subreddit = reddit.subreddit(ccsr)
    subreddit_polarity = analyze_subreddit(subreddit)
    print('{0} | {1}'.format(ccsr, subreddit_polarity))

def analyze_subreddit(subreddit):
  polarity = 0
  for submission in subreddit.top(limit=10):
    polarity += 0.1 * analyze_submission(submission)
  return polarity

def analyze_submission(submission):
  polarity = 0.1 * analyze_sentence(submission.title)

  aggregate_comment_score = 0
  submission.comments.replace_more(limit=0)
  top_level_comments = submission.comments.list()
  for comment in top_level_comments:
    scaling_factor = 1.0 / len(top_level_comments)
    aggregate_comment_score += scaling_factor * analyze_comment(comment)
  polarity += 0.9 * aggregate_comment_score

  return polarity

def analyze_comment(comment):
  polarity = analyze_sentence(comment.body)
  return polarity

def analyze_sentence(sentence):
  vader_polarity_scores = vader_analyzer.polarity_scores(sentence)
  vader_polarity = vader_polarity_scores['pos'] - vader_polarity_scores['neg']
  return vader_polarity

if __name__ == '__main__':
  main()