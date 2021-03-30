from psaw import PushshiftAPI
import datetime as dt
import pandas as pd

api = PushshiftAPI()

start_epoch = int(dt.datetime(2020, 12, 20).timestamp())
end_epoch = int(dt.datetime(2021, 1, 21).timestamp())

gen = api.search_comments(
    after=start_epoch,
    before=end_epoch,
    subreddit='stellar',
    filter=['author', 'body', 'created_utc', 'score']
)

df = pd.DataFrame([thing.d_ for thing in gen])

df['created_at'] = df['created_utc'].apply(lambda x: dt.datetime.fromtimestamp(x))
df.to_csv('reddit_data.csv')
