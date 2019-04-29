title: Automatic activity tracking with Google Sheets, PeriodIndex, and matplotlib
slug: activity-tracking
category: dataviz
date: 2019-04-29
modified: 2019-04-29


There's a lot to keep track of during the job search process, and last week I decided it would be nice to have an easy, automatic way to visualize my job search activity from week to week.

I've been tracking my efforts using Google Sheets, and wanted to be able to get an updated visualization drawing directly from the online worksheet by running a script. After a morning of work, I found a solution that would allow me to use Google Sheets as a database, and get updated visualizations using Python and `matplotlib`.

Here is the result!

![Job Search Image]({static}/images/2019-04-26-report.png)

If you would like to set up something similar for yourself, _read on._

## Using Google Sheets as a Database

In order to use Google Sheets to track your activities you'll need:

1. OAuth2 credentials from the Google Developers Console. [You may find a guide for this part by clicking here.]()
2. The `gspread` library in order to get the updated activity tracking worksheet from Google Sheets as a pandas DataFrame

To follow along with this example, the tracking sheet itself is only required to have two columns:

* `Date` - the date the job search activity took place
* `Action` - the kind of action taken

## Importing Libraries

This first code cell imports the necessary libraries, including `datetime` which we will need in order to automatically timestamp our plot's filename, and sets the global style for the final plot.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread
import datetime

from oauth2client.service_account import ServiceAccountCredentials

plt.style.use('seaborn')
```

## Connecting to Google Sheets

This cell connects to the Google Drive API using the `gspread` Client object. `activity_sheet` contains the exact name of the worksheet that contains the `Date` and `Action` columns.

```python
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('../data/job_search_creds.json', scope)
gc = gspread.authorize(creds)
activity_sheet = "Name of Activity Tracking Sheet"
```

## Retrieving Activity Worksheet

This function uses the `gspread` Client to retrieve the activity worksheet and converts it into a pandas DataFrame.

```python
def get_activity(gc, name):
    """Returns worksheet from connected workbook.
    
    Args:
        gc (Client): gspread client object
        worksheet (str): Name of worksheet
    Returns:
        df (DataFrame): Worksheet as pandas DataFrame
    """
    book = gc.open(name)
    job_search = book.get_worksheet(1)
    js_vals = job_search.get_all_values()
    df = pd.DataFrame(js_vals[1:], columns=js_vals[0])
    return df
```

## Using a Period Index

This function does a few important things that bear some explanation:

1. It converts the `Date` column to the `datetime` data type, and sets it as the index of the DataFrame
2. It then converts the the `DatetimeIndex` to a [PeriodIndex](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.PeriodIndex.html), which allows us to group our actions by week rather than by individual day when we pass in `freq="W"`
3. It converts our `Action` column into dummy variables for each category of action, and then sums the counts of each category by week

The nice thing about building the function this way, is that if you add a new kind of action -- say you decide you want to start tracking `Blog Post` as a category so you know how many posts you publish weekly -- this will automatically update without you having to write any new code.

```python
def get_weekly_counts(df, target, dates="Date"):
    """Gets weekly counts of categorical column in DataFrame containing a column of dates.
    
    Args:
        df (DataFrame): DataFrame containing target and date columns
        target (str): Column containing categories to be counted
        dates (str): Column of dates
    Returns:
        counts (DataFrame): DataFrame of weekly counts
    """
    df = df.set_index(pd.to_datetime(df.Date)).drop("Date", axis=1)
    weekly = df.to_period(freq="W")
    actions = pd.get_dummies(weekly[target])
    counts = actions.groupby(actions.index).sum()
    return counts
```

The returned `counts` DataFrame ends up looking like this:

![Counts Example]({static}/images/counts-example.PNG)

## Making the Plot

At long last, we are ready to produce and save the final plot. We make the stacked area plot using the `.plot.area()` pandas DataFrame method, and then customize it using `matplotlib`.

Thanks to `datetime.datetime.now()`, we have an automatic naming scheme for our saved files, which both keeps us from overwriting past plots and makes it easy to see in-folder how things have progressed over time.

```python
def plot_report(counts):
    """Plots job search activity and saves .png to reports folder.
    
    Args:
        counts (DataFrame): DataFrame of activity counts by period
    """
    # Get current date for filename
    now = datetime.datetime.now()
    time_string = now.strftime("%Y-%m-%d")
    
    counts.plot.area()
    plt.title("Job Search Dashboard")
    plt.ylabel("# Actions Taken")
    plt.xlabel("Week")
    plt.tight_layout()
    # These next two lines mark when I personally started my job search
    # You can remove them without consequence
    plt.axvline('4/1/2019', c='r')
    plt.text('4/1/2019', 10, "  Job search start", fontdict={'size':10})
    plt.savefig('../reports/{}-report.png'.format(time_string))
    pass
```

## Calling the Functions

Thanks to our hard work from earlier, all we need to do to generate a new plot which automatically incorporates any changes we've made to our Google Sheet is run the following three lines of code:

```python
df = get_activity(gc, activity_sheet)
counts = get_weekly_counts(df, "Action")
plot_report(counts)
```
![Job Search Image]({static}/images/2019-04-26-report.png)

## Celebration Time

And there you have it! A nice plot that displays weekly activity counts, drawing updates from Google Sheets. This workflow can be used to make any kind of plot using Google Sheets as a kind of lightweight database. Experiment, edit, and enjoy!