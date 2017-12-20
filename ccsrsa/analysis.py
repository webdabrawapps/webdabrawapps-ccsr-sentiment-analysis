
__version__ = '0.0.2'

import praw
import json
import numpy as np
import logging
import functools

from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pprint import pprint
from constants import *

vader_analyzer = SentimentIntensityAnalyzer()

def main():
  '''
  Entry to the program.
  '''
  logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',level=logging.INFO)

  reddit_config = json.load(open(PRAW_CONFIG_FILE))
  reddit = praw.Reddit(**reddit_config)

  subreddit_polarities = {}

  for ccsr in CRYPTOCURRENCY_SUBREDDITS:
    subreddit = reddit.subreddit(ccsr)
    subreddit_polarities[subreddit] = analyze_subreddit(subreddit)

  pprint(subreddit_polarities)

b = np.power(ONE_MONTH_WEIGHT_DEPRECIATION, -1 / ONE_MONTH_DAYS)
a = 1 / np.power(b, MAX_ELAPSED_DAYS)
def _get_submission_ranking(sbm):
  '''
  Returns the submission's ranking/score.

  :param sbm: The praw submission object
  :return: How much this submission should affect the polarity of the subreddit
  '''
  days_since_epoch = (datetime.fromtimestamp(sbm.created) - EPOCH).total_seconds() / 86400
  time_factor = a * b ** days_since_epoch
  score_factor = np.log10(max(1, sbm.score))
  return score_factor * time_factor

def _get_comment_ranking(cmt):
  '''
  Returns the comment's ranking/score.

  :param sbm: The praw comment object
  :return: How much this comment should affect the polarity of its parent submission
  '''
  return max(0, cmt.score)

def analyze_subreddit(subreddit):
  '''
  Determines the polarity of the given subreddit.
  The polarity of a subreddit is given by a weighted average of the polarities
  of its top {NUM_HOT_SUBMISSIONS} submissions.

  :param subreddit: The praw subreddit object
  :return: The polarity of the subreddit
  '''
  polarity = 0

  submissions = set(subreddit.hot(limit=NUM_HOT_SUBMISSIONS))
  submission_rankings = {sbm : _get_submission_ranking(sbm) for sbm in submissions}
  ranking_sum = sum(submission_rankings.values())

  logging.info('-' * 60)
  logging.info('For /r/{0}, looking at {1} hot & new submissions.'.format(subreddit.display_name, len(submissions)))

  for submission, submission_ranking in submission_rankings.items():
    submission_scaling_factor = submission_ranking / ranking_sum
    logging.info('-' * 60)
    logging.info('\'{0}\' '.format(submission.title))
    logging.info('Posted on {0} with {1} net upvotes.'.format(datetime.fromtimestamp(submission.created), submission.score))
    logging.info('Accounting for {0:.2f}% of polarity score for /r/{1}.'.format(submission_scaling_factor * 100, subreddit))
    submission_polarity = analyze_submission(submission)
    logging.info("Net polarity for post: {0:.2f}".format(submission_polarity))
    polarity += submission_scaling_factor * submission_polarity

  return polarity

def analyze_submission(submission):
  '''
  Determines the polarity of the given submission.
  The polarity of a submission is given by a weighted average of the polarities
  of all of its child comments and its title.

  :param submission: The praw submission object
  :return: The polarity of the submission
  '''
  polarity = analyze_sentence(submission.title)

  submission.comments.replace_more(limit=0)

  if submission.comments:
    top_level_comments = submission.comments.list()
    top_level_comment_rankings = {cmt: _get_comment_ranking(cmt) for cmt in top_level_comments}
    ranking_sum = sum(top_level_comment_rankings.values())

    if ranking_sum:
      polarity *= SUBMISSION_TITLE_WEIGHT

      aggregate_comment_score = 0
      for comment, comment_ranking in top_level_comment_rankings.items():
        comment_scaling_factor = comment_ranking / ranking_sum
        aggregate_comment_score += comment_scaling_factor * analyze_comment(comment)

      polarity += SUBMISSION_COMMENTS_WEIGHT * aggregate_comment_score

  return polarity

def analyze_comment(comment):
  '''
  Determines the polarity of the given comment.
  The polarity of a comment is given by the polarity of its body added to the
  polarity of any children it might have, weighted accordingly.

  :param comment: The praw comment object
  :return: The polarity of the comment
  '''
  polarity = analyze_sentence(comment.body)

  comment.replies.replace_more(limit=0)

  if comment.replies:
    replies = comment.replies.list()
    reply_rankings = {cmt: _get_comment_ranking(cmt) for cmt in replies}
    ranking_sum = sum(reply_rankings.values())

    if ranking_sum:
      polarity *= COMMENT_BODY_WEIGHT

      aggregate_reply_score = 0
      for reply, reply_ranking in reply_rankings.items():
        reply_scaling_factor = reply_ranking / ranking_sum
        aggregate_reply_score += reply_scaling_factor * analyze_comment(reply)

      polarity += COMMENT_CHILDREN_WEIGHT * aggregate_reply_score

  return polarity

@functools.lru_cache(maxsize=None)
def analyze_sentence(sentence):
  '''
  Determines the polarity of the given text. Uses the VADER sentiment
  analyzer. TODO: Consider using other sentiment analyzers.

  :param sentence: The sentence to analyze
  :return: The polarity of the sentence (-1 for negativity, +1 for positivity)
  '''
  sentiments = []

  vader_polarity_scores = vader_analyzer.polarity_scores(sentence)
  sentiments.append(vader_polarity_scores['pos'] - vader_polarity_scores['neg'])

  return np.mean(sentiments)

if __name__ == '__main__':
  main()