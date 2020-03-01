import pickle
from collections import namedtuple
from datetime import datetime, timedelta

SplittedDate = namedtuple("SplittedDate", ["date", "weekday", "hour"])


def get_date(s):
    return s.split(" ")[0]


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return (d2 - d1).days


def plus_days(d1, n_days: int = 0):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    return (d1 + timedelta(days=n_days)).strftime("%Y-%m-%d")


def split_date(s) -> SplittedDate:
    d = None
    if "T" in s:
        d = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    else:
        d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return SplittedDate(d.strftime("%Y-%m-%d"), d.weekday(), d.hour)