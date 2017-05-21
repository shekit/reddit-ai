# expects 3 arguments in when called - interval, gointopast and starttime

# scrape 6 months - 15552000
# interval of 2 hours - 7200
# start time - 24th mar 2017 midnight - 1490313600, Sep 20 2016 - 1474329600, Mar 20 2016 - 1458432000

# 5 hours 17 minute to scrape these 6, only titles for 6 months of posts

import json
import os
import time
import praw
import sys

subreddits = [
			"askreddit",
			"todayilearned",
			"writingprompts",
			"lifeprotips",
			"explainlikeimfive",
			"showerthoughts"
			]

num_subs = 0

class ScrapeReddit:

	def __init__(self, subreddit=None, interval=7200, gointopast=2592000, start_time=None):
		self.postCounter = 0
		self.commentCounter = 0

		self.interval = interval					# increments to move back in time
		self.gointopast = gointopast				# how far back to go in time

		if start_time is None:
			self.start_time = int(time.time())     # time to start going backward from
		else:
			self.start_time = start_time

		self.end_timestamp = start_time - gointopast  # time to stop scraping
		self.subreddit = subreddit 					# subreddit to scrape

		if not os.path.exists(subreddit):			# make folder to store data
			os.mkdir(subreddit)

		self.reddit = praw.Reddit(client_id='Bc805C9xmKhIVg',
						client_secret='foiyZK35ZPMopQ_TRZvRlhK6jAM',
						user_agent='User-agent: ubuntu:Bc805C9xmKhIVg:v0.1 (by /u/abhi3188)')

		

		

	def save_submission(self, data):
		with open(os.path.join(self.subreddit, self.subreddit+".txt"), 'a') as f:
			
			for item in data:
				self.postCounter+=1
				f.write(json.dumps({
					"title":item.title,
					"score":item.score
					}))
				f.write(",") #\n
			#print("Saved Titles: {}".format(self.postCounter))
			f.close()

		with open(os.path.join(self.subreddit, self.subreddit+'_comm.txt'), 'a') as f:
			
			for item in data:
				item.comments.replace_more(limit=0)
				for comment in item.comments:
					self.commentCounter+=1
					f.write(json.dumps({"body":comment.body,
									"score":comment.score}))
					f.write(",") #\n
			#print("Saved Comments: {}".format(self.commentCounter))
			f.close()

	def download_subreddit(self):
		global num_subs

		start_timestamp = self.start_time

		past_timestamp = start_timestamp - self.interval

		while True:
			try:
				sub = self.reddit.subreddit(self.subreddit)
				search_results = list(sub.submissions(past_timestamp, start_timestamp))

			except Exception as e:
				print("Exception: ",e)
				time.sleep(600)
				continue

			#print("Got {} submissions in interval {}..{}".format(len(search_results), past_timestamp, start_timestamp))

			self.save_submission(search_results)
			
			#go back further in time in by whatever interval is set
			start_timestamp = past_timestamp
			past_timestamp = start_timestamp - self.interval

			if past_timestamp < self.end_timestamp:
				print("FINISHED SCRAPING: {}".format(self.subreddit))
				print("No. Posts: {}".format(self.postCounter))
				print("No. Comments: {}".format(self.commentCounter))
				num_subs+=1
				scrapeSub()
				return
				break

def scrapeSub():
	global num_subs
	global subreddits

	if num_subs < len(subreddits):
		print("SCRAPING: {}".format(subreddits[num_subs]))
		scraper = ScrapeReddit(subreddit=subreddits[num_subs], interval=int(sys.argv[1]), gointopast=int(sys.argv[2]), start_time=int(sys.argv[3]))
		scraper.download_subreddit()

	else:
		print("ALL SUBS SCRAPED!")
		print("END TIME: {}".format(int(time.time())))
		return


def main():
	print("START TIME: {}".format(int(time.time())))
	scrapeSub()

if __name__ == "__main__":
	main()

# 2 days of posts from funny take roughly 10 minutes with this script

