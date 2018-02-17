import tweepy
import time
import os
import sys
import json
import csv
import argparse
import random
from datetime import datetime
from twitter import *
from twitter import follow as f

USER_DIR = 'twitter-users'
TWEET_DIR = 'user-tweets'
MAX_FOLLOWERS = 50000
MAX_TWEETS = 1000
MAX_FOLLOWING = 1000

# Create the directories we need
if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)

if not os.path.exists(TWEET_DIR):
    os.makedirs(TWEET_DIR)

enc = lambda x: x.encode('utf-8', errors='ignore')

# == Consumer and access keys ==

# One
# CONSUMER_KEY    = 'gATryKHssIvZBLvih1eBqiWIa'
# CONSUMER_SECRET = 'cXmja5S2eXKXRTwyyWY41aLCYPwhdZfp0H6zZ3CDyyfLz5IRp3'
# ACCESS_TOKEN    = '912588167043264512-7W9yGob6iMOawZGdUqFFutcORvFEm18'
# ACCESS_TOKEN_SECRET = 'Z7Jjfu9Q6nnb41ItgJr8FUyd64gS2a2fprM0IitEvx3cS' 

#Two
# CONSUMER_KEY    = 'G7sdeQo6xuXkywxbi6odKPalE' 
# CONSUMER_SECRET = 'PYhsHiPS5Ruf0LQXmcQNTDTvcdGQFGLd419JECgOlj505H1OUc'
# ACCESS_TOKEN    = '912588167043264512-8hQjCeNFGYhZOgGDRqIV4FynV9QWh2D'
# ACCESS_TOKEN_SECRET = 'vXBWD0eCakcAHxMSs4cS8dWvmkpFXe1H4X0ncwxWfF1et'

#Three
# CONSUMER_KEY    = 'xwhTaFK7UmVVVXNI546E7nUNE' 
# CONSUMER_SECRET = '55PQb8W8Yx6ljDSNdP0Pp44YKp1eaw4J5gfO8SN8bRBX4BUbqz'
# ACCESS_TOKEN    = '912588167043264512-kWOQYM4ORFoKgC9M7kFqioZEwmR6aj8'
# ACCESS_TOKEN_SECRET = 'Clc6bsIMCQvMah1CAftSFSFOhIWrYwHvVDOKmnN8WyGTJ'

#Four
# CONSUMER_KEY    = 'CVwDtlqU4q1ylUwsZu9bAQKHH'
# CONSUMER_SECRET = 'ZQ4KrbfzZoVdPfrqgAemsJKUFHvqhCDeohwly3kjzt1vJBy6Zk'
# ACCESS_TOKEN    = '912588167043264512-7JES86OJv4VQp3EdrYE9XL0Qc5XOId9'
# ACCESS_TOKEN_SECRET = 'g1cZOSiQjsmDOOwRJDbXUJCDVSd9iK73lt1CssLAlGfGU'

#Five
# CONSUMER_KEY    = 'vblohBV2jndZVzROO9qR6wP8d'
# CONSUMER_SECRET = 'XjyVZ9B6ij560YnVz9Rs0ZDG44vnQ85KG3XmQUUzDUU6A8oLTS'
# ACCESS_TOKEN    = '912588167043264512-cfjSQ9tuXsloghUZOsb9AUz6b39hS6Q'
# ACCESS_TOKEN_SECRET = 'SQCkT7A6gOmOGTrs2hv9rIt0G4rvt4NkXEh9WHsueR94T'

#Six
CONSUMER_KEY    = 'ChuECwSgVFTuK7aLCqQnZGZjs'
CONSUMER_SECRET = 'ajHAAWtLi0MpgU0XKT8AMp3hTuq1y60Fvwlzcea84vNJlUlawB'
ACCESS_TOKEN    = '912588167043264512-YnQfvGpqASPK9GjaLCrPSwNRxfKKfqf'
ACCESS_TOKEN_SECRET = 'Dvx8XLQn6EAw9bhtFBLutZ01837yF613nx59aOlwkN0Cc'

# == OAuth Authentication ==

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

##########################################

def get_follower_ids(t, centre, max_depth=1, current_depth=0, already_seen=[]):

    if current_depth == max_depth:
        print 'out of depth'
        return already_seen

    if centre in already_seen:
        # we've been here before
        print 'Already been here.'
        return already_seen
    else:
        already_seen.append(centre)

    try:
        userfname = os.path.join(USER_DIR, str(centre) + '.json')
        if not os.path.exists(userfname):
            print 'Retrieving user details for twitter id %s' % str(centre)
            # Getting current user info
            user = t.users.lookup(user_id=centre)[0]

            # Getting user follower ids
            screen_name = api.lookup_users(user_ids=[centre])[0].screen_name
            follower_ids = f.follow(t, screen_name)

            # Also get IDs of users this user is following
            following_ids = f.follow(t, screen_name, followers=False)

            # user = api.get_user(centre)

            d = {'name': user['name'],
                 'screen_name': user['screen_name'],
                 'profile_image_url' : user['profile_image_url'],
                 'created_at' : str(user['created_at']),
                 'id': user['id'],
                 'friends_count': user['friends_count'],
                 'followers_count': user['followers_count'],
                 'followers_ids': follower_ids,
                 'following_ids': following_ids}

            if 'withheld_in_countries' in user:
                d['withheld_in_countries'] = user['withheld_in_countries']

            with open(userfname, 'w') as outf:
                outf.write(json.dumps(d, indent=1))

        else:
            user = json.loads(file(userfname).read())
            follower_ids = user['followers_ids']

        get_all_tweets(t, user['screen_name'], user['id'])
        screen_name = enc(user['screen_name'])

        # get followers of followers
        cd = current_depth
        if cd+1 < max_depth:
            for fid in follower_ids[:MAX_FOLLOWERS]:
                already_seen = get_follower_ids(t, fid, max_depth=max_depth,
                    current_depth=cd+1, already_seen=already_seen)

        if cd+1 < max_depth and len(follower_ids) > MAX_FOLLOWERS:
            print 'Not all followers retrieved for %s.' % screen_name

    except Exception, error:
        print 'Error retrieving follower for user id: ', centre
        print error
        return []
    return already_seen

def get_all_tweets(t, screen_name, user_id, max_id=None):
    fname = os.path.join(TWEET_DIR, str(screen_name) + '_tweets.csv')
    if not os.path.exists(fname):
        #initialize a list to hold all the tweepy Tweets
        alltweets = []

        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name = screen_name,count=200)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #keep getting tweets until there are no tweets left
        while len(new_tweets) > 0 and len(alltweets) < MAX_TWEETS:
            print "getting tweets before %s" % (oldest)
            
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
            
            #save most recent tweets
            alltweets.extend(new_tweets)
            
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
            
            print "...%s tweets downloaded so far" % (len(alltweets))
        
        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), tweet.retweet_count,
                    tweet.favorite_count] for tweet in alltweets]
        
        #write the csv
        with open(fname, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["id","created_at","text","retweet_count","favorite_count"])
            writer.writerows(outtweets)

    ############################################################
    # One of the ways to get tweets, but I think we only get 200

    # kwargs = dict(count=3200, screen_name=screen_name)
    # if max_id:
    #     kwargs['max_id'] = max_id

    # n_tweets = 0
    # tweets = t.statuses.user_timeline(**kwargs)
    # # for tweet in tweets:
    # #     if tweet['id'] == max_id:
    # #         continue
    # #     print("%s %s\nDate: %s" % (tweet['user']['screen_name'],
    # #                                tweet['id'],
    # #                                tweet['created_at']))
    # #     if tweet.get('in_reply_to_status_id'):
    # #         print("In-Reply-To: %s" % tweet['in_reply_to_status_id'])
    # #     print()
    # #     for line in tweet['text'].splitlines():
    # #         printNicely('    ' + line + '\n')
    # #     print()
    # #     print()
    # #     max_id = tweet['id']
    # #     n_tweets += 1
    # # return n_tweets, max_id

    # return tweets

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--screen-name", required=True, help="Screen name of twitter user")
    ap.add_argument("-d", "--depth", required=True, type=int, help="How far to follow user network")
    args = vars(ap.parse_args())

    twitter_screenname = args['screen_name']
    depth = int(args['depth'])

    if depth < 1 or depth > 5:
        print 'Depth value %d is not valid. Valid range is 1-5.' % depth
        sys.exit('Invalid depth argument.')

    print 'Max Depth: %d' % depth
    matches = api.lookup_users(screen_names=[twitter_screenname])

    t = Twitter(auth=OAuth(ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
    print get_follower_ids(t, matches[0].id, max_depth=depth)
