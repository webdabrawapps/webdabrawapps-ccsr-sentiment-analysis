
import praw
import nltk
import json
import numpy as np
import logging

from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from constants import *

vader_analyzer = SentimentIntensityAnalyzer()

def main():
  logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',level=logging.INFO)

  reddit_config = json.load(open(PRAW_CONFIG_FILE))
  reddit = praw.Reddit(**reddit_config)

  subreddit_polarities = {}

  for ccsr in CRYPTOCURRENCY_SUBREDDITS:
    subreddit = reddit.subreddit(ccsr)
    subreddit_polarities[subreddit] = analyze_subreddit(subreddit)

  logging.info(subreddit_polarities)

def analyze_subreddit(subreddit):

  polarity = 0
  epoch = datetime(2009, 1, 3, 12, 0, 0)

  def submission_ranking(sbm):
    if not sbm.score:
      return 0
    score_factor = np.log10(max(1, abs(sbm.score)))
    time_factor = (1 + 1e-7) ** (datetime.fromtimestamp(sbm.created) - epoch).total_seconds() / 1e11
    return score_factor + time_factor

  top_submissions = subreddit.hot(limit=NUM_SUBMISSIONS_TO_CONSIDER)
  top_submission_rankings = {sbm : submission_ranking(sbm) for sbm in top_submissions}
  ranking_sum = sum(top_submission_rankings.values())

  for submission, submission_ranking in top_submission_rankings.items():
    submission_scaling_factor = submission_ranking / ranking_sum
    logging.info('Submission \'{0}\' posted on {1} with {2} net upvotes.'.format( \
      submission.title, datetime.fromtimestamp(submission.created), submission.score))
    logging.info('Accounting for {0:.2f}% of polarity score for /r/{1}.'.format(submission_scaling_factor * 100, subreddit))
    polarity += submission_scaling_factor * analyze_submission(submission)

  return polarity

def analyze_submission(submission):
  polarity = SUBMISSION_TITLE_WEIGHT * analyze_sentence(submission.title)

  aggregate_comment_score = 0
  submission.comments.replace_more(limit=0)
  top_level_comments = submission.comments.list()
  for comment in top_level_comments:
    scaling_factor = 1.0 / len(top_level_comments)
    aggregate_comment_score += scaling_factor * analyze_comment(comment)
  polarity += SUBMISSION_COMMENTS_WEIGHT * aggregate_comment_score

  return polarity

def analyze_comment(comment):
  polarity = analyze_sentence(comment.body)
  return polarity

def analyze_sentence(sentence):
  sentiments = []

  vader_polarity_scores = vader_analyzer.polarity_scores(sentence)
  sentiments.append(vader_polarity_scores['pos'] - vader_polarity_scores['neg'])

  sentiments.append(TextBlob(sentence).sentiment.polarity)

  return np.mean(sentiments)

if __name__ == '__main__':
  main()