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

enc = lambda x: x.encode('ascii', errors='ignore')

# The consumer keys can be found on your application's Details
# page located at https://dev.twitter.com/apps (under "OAuth settings")

# The access tokens can be found on your applications's Details
# page located at https://dev.twitter.com/apps (located
# under "Your access token")

CONSUMER_KEY    = '****************'
CONSUMER_SECRET = '****************'
ACCESS_TOKEN    = '****************'
ACCESS_TOKEN_SECRET = '****************'

# == OAuth Authentication ==
#
# This mode of authentication is the new preferred way
# of authenticating with Twitter.
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

##########################################

def get_following_ids(t, centre, max_depth=1, current_depth=0, taboo_list=[]):

    if current_depth == max_depth:
        print 'out of depth'
        return taboo_list

    if centre in taboo_list:
        # we've been here before
        print 'Already been here.'
        return taboo_list
    else:
        taboo_list.append(centre)

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
            following_ids = user['following_ids']

        get_all_tweets(t, user['screen_name'])
        screen_name = enc(user['screen_name'])
        # fname = os.path.join(FOLLOWER_DIR, screen_name + '.csv')
        # followerids = []

        ### Writes CSV file for the followers. Currently commented out because I only need the iDs

        # if not os.path.exists(fname):
        #     print 'No cached data for screen name "%s"' % screen_name
        #     with open(fname, 'w') as outf:
        #         params = (enc(user['name']), screen_name)
        #         print 'Retrieving followers for user "%s" (%s)' % params

        #         # page over followers
        #         # c = tweepy.Cursor(api.followers, id=user['id']).items()

        #         follower_count = 0
        #         if user_ids:
        #             users = f.lookup(t, user_ids) # These are full user objects

        #         print "starting for loop"
        #         print users
        #         for i, username in users.iteritems():
        #             followerids.append(i)
        #             params = (i, enc(username))
        #             outf.write('%s\t%s\n' % params)
        #             follower_count += 1
        #             if follower_count >= MAX_FOLLOWERS:
        #                 print 'Reached max no. of followers for "%s".' % str(username)
        #                 break
        # else:
        #     followerids = [int(line.strip().split('\t')[0]) for line in file(fname)]

        # print 'Found %d followers for %s' % (len(followerids), screen_name)

        # get followers of followers
        cd = current_depth
        if cd+1 < max_depth:
            for fid in following_ids[:MAX_FOLLOWING]:
                taboo_list = get_following_ids(t, fid, max_depth=max_depth,
                    current_depth=cd+1, taboo_list=taboo_list)

        if cd+1 < max_depth and len(following_ids) > MAX_FOLLOWING:
            print 'Not all following retrieved for %s.' % screen_name

    except Exception, error:
        print 'Error retrieving following for user id: ', centre
        print error

        # if fname and os.path.exists(fname):
        #     os.remove(fname)
        #     print 'Removed file "%s".' % fname

        # sys.exit(1)
        return []
    return taboo_list

def get_all_tweets(t, screen_name, max_id=None):
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

        #keep grabbing tweets until there are no tweets left to grab
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
    print get_following_ids(t, matches[0].id, max_depth=depth)
