"""Market clock: clock/jobs.json ke hisaab se is repo ke workflows time pe chalata hai.

GitHub ka apna cron schedule aksar late ya miss hota hai. Ye ek lamba job hai jo
har 15 sec time dekhta hai aur sahi minute pe workflow_dispatch karta hai.
Usage: python clock/market_clock.py HH:MM   (IST, is time tak chalega)
jobs.json: [{"wf": "x.yml", "at": ["14:41"]}  ya  {"wf": "y.yml", "every": 15, "from": "09:15", "to": "15:30"}]
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
HERE = os.path.dirname(os.path.abspath(__file__))


def mins(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def due(job, m):
    if "at" in job:
        return any(m == mins(t) for t in job["at"])
    a, b = mins(job["from"]), mins(job["to"])
    return a <= m <= b and (m - a) % job["every"] == 0


def dispatch(wf):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/workflows/{wf}/dispatches",
        data=json.dumps({"ref": "main"}).encode(), method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.read()[:200]!r}"
    except Exception as e:
        return f"error {e}"


def main():
    end = mins(sys.argv[1])
    jobs = json.load(open(os.path.join(HERE, "jobs.json")))
    if datetime.now(IST).weekday() >= 5:
        print("Aaj Saturday/Sunday hai, kuch nahi chalana.")
        return
    fired = set()
    while True:
        now = datetime.now(IST)
        m = now.hour * 60 + now.minute
        if m >= end:
            print(f"{now:%H:%M} IST: clock khatam.")
            return
        for j in jobs:
            key = (j["wf"], m)
            if key not in fired and due(j, m):
                fired.add(key)
                print(f"{now:%H:%M} IST  {j['wf']} -> {dispatch(j['wf'])}", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
