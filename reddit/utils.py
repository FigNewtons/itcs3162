import praw
import datetime
from bs4 import BeautifulSoup

# Configuration for PRAW -- the user must supply this info manually
from config import *

# Initialize PRAW instance for global use
reddit = praw.Reddit(USER_AGENT)
reddit.set_oauth_app_info(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)


def parse_comment(comment):
    """Return the links and quotes from a thread comment. """

    comment_html = comment.body_html
    soup = BeautifulSoup(comment_html, 'html.parser')

    links = [ link.get('href') for link in soup.find_all('a')]
    quotes = [quote.get_text().replace('\n','') for quote in soup.find_all('blockquote')]

    return (links, quotes)


def get_comment_tree(thread):
    """Return the (summarized) comment tree of a given thread. """

    thread.replace_more_comments(limit = None, threshold = 0)
    op = "" if not thread.author else thread.author.name

    tree = build_tree(thread.comments, op)

    return tree


def build_tree(comment_tree, op):
    """Helper function to build comment tree. """
    tree = []

    for parent in comment_tree:

        author = parent.author
        time = praw.helpers.time.localtime(parent.created_utc)
        comment_length = len(parent.body)
        score = parent.score
        gold = parent.gilded
        edited = parent.edited
        is_op = False if not author else author.name == op.name

        replies = get_comment_tree(parent.replies, op)

        comment = {'author': author, 'time': time, 
                   'comment_length': comment_length, 
                   'score': score, 'gold': gold, 
                   'edited': edited, 'is_op': is_op, 
                   'replies' : replies}
    
        tree.append(comment)

    return tree


def get_commenters(thread, attr = 'freq'):
    """Return a tuple containing three items:
            1. Dictionary of usernames and corresponding attribute value
            2. Set of commenter usernames
            3. List of dict.items() from (1) ranked by frequency

        - thread is a Submission object
        - attr is a string within set {'freq', 'karma', 'gold', 'time'};
            This determines what values are put into dictionary
            and consequently sorted.

            freq  = total number of comments within thread
            karma = total karma count across posts
            gold  = total gold count across posts
            time  = timestamp of most recent post
    """
    func = { 
        'freq' : lambda comment, val, exists: 1,
        'karma': lambda comment, val, exists: comment.score,
        'gold' : lambda comment, val, exists: comment.gilded,
        'time' : lambda comment, val, exists: max(comment.created_utc - val, 0) if exists else comment.created_utc
    }

    thread.replace_more_comments(limit = None, threshold = 0)
    comments = praw.helpers.flatten_tree(thread.comments)

    commenters = {}
    for comment in comments:
        if comment.author is not None:  
            name = comment.author.name
            if name in commenters:
                commenters[name] += func[attr](comment, commenters[name], True)
            else:
                commenters[name] = func[attr](comment, None, False)

    ranks = sorted(list(commenters.items()), key= lambda x: x[1], reverse=True)

    # commenters, names, ranks
    return (commenters, set(commenters), ranks)


if __name__ == '__main__':

    math_thread = reddit.get_submission(submission_id = '49cj33')
    comment_tree = get_comment_tree(math_thread)

    print(comment_tree)

    print("\n\nop: {0}\n".format(math_op))

    print(thread_stats(math_thread))
    

