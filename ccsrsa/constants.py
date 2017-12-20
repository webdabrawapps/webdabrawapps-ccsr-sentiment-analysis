
from datetime import datetime
import numpy as np

PRAW_CONFIG_FILE = 'config.json'
CRYPTOCURRENCY_SUBREDDITS = [
  'Augur',
  'ArkEcosystem',
  'Bitcoin',
  'BitcoinCash',
  'DashPay',
  'Ethereum',
  'GolemProject',
  'IOTA',
  'Litecoin',
  'Monero',
  'omise_go',
  'Ripple'
]

EPOCH = datetime(2009, 1, 3, 12, 0, 0) # Bitcoin creation date!
MAX_ELAPSED_DAYS = (datetime.now() - EPOCH).total_seconds() / 86400
ONE_MONTH_DAYS = 31
ONE_MONTH_WEIGHT_DEPRECIATION = 0.1

NUM_HOT_SUBMISSIONS = 10

SUBMISSION_TITLE_WEIGHT = 0.3
SUBMISSION_COMMENTS_WEIGHT = 1 - SUBMISSION_TITLE_WEIGHT

COMMENT_BODY_WEIGHT = 0.7
COMMENT_CHILDREN_WEIGHT = 1 - COMMENT_BODY_WEIGHT