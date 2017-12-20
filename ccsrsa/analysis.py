
__version__ = '0.0.1'

import praw
import json
import numpy as np
import logging
import functools

from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pprint import pprint
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

  pprint(subreddit_polarities)

def _get_submission_ranking(sbm):
  if not sbm.score:
    return 0
  epoch = datetime(2009, 1, 3, 12, 0, 0) # Bitcoin creation date!
  score_factor = np.log10(max(1, abs(sbm.score)))
  time_factor = (1 + 1e-7) ** (datetime.fromtimestamp(sbm.created) - epoch).total_seconds() / 1e11
  return score_factor + time_factor

def _get_comment_ranking(cmt):
  return cmt.score

def analyze_subreddit(subreddit):

  polarity = 0

  top_submissions = subreddit.hot(limit=NUM_SUBMISSIONS_TO_CONSIDER)
  top_submission_rankings = {sbm : _get_submission_ranking(sbm) for sbm in top_submissions}
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

  submission.comments.replace_more(limit=0)
  top_level_comments = submission.comments.list()
  top_level_comment_rankings = {cmt: _get_comment_ranking(cmt) for cmt in top_level_comments}
  ranking_sum = sum(top_level_comment_rankings.values())
  aggregate_comment_score = 0

  for comment, comment_ranking in top_level_comment_rankings.items():
    comment_scaling_factor = comment_ranking / ranking_sum
    aggregate_comment_score += comment_scaling_factor * analyze_comment(comment)

  polarity += SUBMISSION_COMMENTS_WEIGHT * aggregate_comment_score

  return polarity

def analyze_comment(comment):
  polarity = analyze_sentence(comment.body)

  if comment.replies:

    comment.replies.replace_more(limit=0)
    replies = comment.replies.list()
    reply_rankings = {cmt: _get_comment_ranking(cmt) for cmt in replies}
    ranking_sum = sum(reply_rankings.values())
    aggregate_reply_score = 0

    if ranking_sum:
      polarity *= COMMENT_BODY_WEIGHT

      for reply, reply_ranking in reply_rankings.items():
        reply_scaling_factor = reply_ranking / ranking_sum
        aggregate_reply_score += reply_scaling_factor * analyze_comment(reply)

      polarity += COMMENT_CHILDREN_WEIGHT * aggregate_reply_score

  return polarity

@functools.lru_cache(maxsize=None)
def analyze_sentence(sentence):
  sentiments = []

  vader_polarity_scores = vader_analyzer.polarity_scores(sentence)
  sentiments.append(vader_polarity_scores['pos'] - vader_polarity_scores['neg'])

  sentiments.append(TextBlob(sentence).sentiment.polarity)

  return np.mean(sentiments)

if __name__ == '__main__':
  main()