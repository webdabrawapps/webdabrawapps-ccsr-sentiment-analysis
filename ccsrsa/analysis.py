
import praw
import nltk
import json

from aes import AESCipher
from constants import *

def main():

  reddit_config = json.load(open(PRAW_CONFIG_FILE))
  reddit = praw.Reddit(**reddit_config)



if __name__ == '__main__':
  main()