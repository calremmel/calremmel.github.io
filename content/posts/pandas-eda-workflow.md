title: Human Readable EDA in Python
slug: human-readable-eda-python
category: dataviz
date: 2019-05-24

# Why Pipes are Great

I've been working with a team whose computational work is done primarily in R. Python is my primary language and I remain much more comfortable with its scientific stack of tools, but as I've gotten to know `ggplot2` and the `tidyverse` ecosystem of tools in R, I've been finding myself wishing that I could come up with something a little closer to that workflow in Python.

One of the major things I like about how code is written within the `tidyverse` is the central importance of the pipe command: `%>%`. In R, just about every transformation you want to apply to data is going to be done through a function. If you want to apply many transformations, this can quickly lead to lots of nested functions, which can be cumbersome to read and unpleasant to edit -- unless you love counting parenthesis!

The beauty of the pipe is that it takes the output of its line and passes it as the first argument into the function that appears on the next line of code. By way of example, let's say you have a DataFrame `finances` with columns `year` and `revenue`, and you want to get the total revenue for each year after 2008. Without the pipe, that code might look like this:

```R
summarize(group_by(filter(finances, year > 2008), year), totalRevenue = sum(revenue))
```

I'd like to emphasize that while this code will run, nobody would ever write it like this. This is only by way of example.

With the pipe, this becomes much more human readable:

```R
finances %>%
  filter(year > 2008) %>%
  group_by(year) %>%
  summarize(totalRevenue = sum(revenue))
```

The syntax for `ggplot2` does not use the pipe, but it uses a similar "one operation on one line" design philosophy that makes it easy to iterate and try adding and removing different visual elements.

Let's say we've saved the result of the previous operation as a new DataFrame called `revenue_by_year`. We would make our chart with the following syntax:

```R
ggplot(revenue_by_year) +
  aes(year, totalRevenue) +
  geom_col()
```

If we wanted to flip the orientation, for example, we would need nothing more than add in a line saying so:

```R
ggplot(revenue_by_year) +
  aes(year, totalRevenue) +
  geom_col() +
  coord_flip()
```

Nice, huh? Finally, if you wanted, you could do all of this in one step using the pipe to pass the summary directly into the `ggplot` function. Thanks to the stepwise design philosophy of the `tidyverse`, it remains readable despite the length.

```R
finances %>%
  filter(year > 2008) %>%
  group_by(year) %>%
  summarize(totalRevenue = sum(revenue)) %>%
  ggplot() +
    aes(year, totalRevenue) +
    geom_col() +
    coord_flip()
```

## Human Readable EDA with Pandas

`Pandas` is a wonderful library for data wrangling and exploration, and includes some great tools for making quick exploratory charts. It doesn't generally have the nested functions issue that we encountered in R code, but it does have another readability challenge: method chaining.

In order to recreate the same summary in Python using `Pandas`, we would need to chain a couple of methods, like this:

```Python
finances.loc[finances["year"] > 2008].groupby("year").sum().reset_index().plot.bar("year", "totalRevenue")
```

This is not ideal for readability. It is better than nesting, as at least the operations are in the same order in which they take place. Still, this is a very long line.

One solution to this is the backslash, which breaks up lines. With that, we could get a `tidyverse` feel by manually indenting each line like so.

```Python
finances \ 
    .loc[finances["year"] > 2008] \
    .groupby("year") \
    .sum() \
    .reset_index() \
    .plot.bar("year", "totalRevenue")
```

I don't know about you, but something about this just doesn't feel right. Does PEP8 have something to say?

*It does.*

```
The preferred way of wrapping long lines is by using Python's implied line continuation inside parentheses, brackets and braces. Long lines can be broken over multiple lines by wrapping expressions in parentheses. These should be used in preference to using a backslash for line continuation.
```

It was news to me that parenthesis had this property in Python! This solves our problem entirely, and has the added bonus of taking care of indentation. Our code becomes this:

```Python
(finances
 .loc[finances["year"] > 2008]
 .groupby("year").sum()
 .reset_index()
 .plot.bar("year", "totalRevenue"))
```

# But Where are the Plots?

To make this all a little less abstract, here's an example of this workflow using the New Hampshire State Budget.

I wanted to plot the median expense for every government agency, in order. There were two major outliers, so I wanted to plot those separately. Here is the code, followed by the output:

```Python
# Get median expense by government agency.
fy_2018_expense_medians = fy_2018_expenses.groupby("agency_name").median()
# Filter outliers and create bar chart.
(fy_2018_expense_medians
 .sort_values("fy_2018_actual_expense", ascending=False)
 .loc[fy_2018_expense_medians.fy_2018_actual_expense < 40000000, "fy_2018_actual_expense"]
 .plot.bar(
     figsize=(10,7), 
     grid=False, 
     title="FY18 Median Expense by Government Agency"
 ))
# Set axis labels.
plt.xlabel("Agency Name")
plt.ylabel("Median Expense in Dollars")
```

![png]({static}/images/median_no_outliers.png)

You will notice that I saved the aggregation as a separate DataFrame before plotting. This is because in order to filter by rows in `Pandas`, I needed to refer to a an actual DataFrame, whereas in R you are able to refer to the dynamically created summary column immediately. Still, this is a small sacrifice.

In order to plot the outliers, all I needed to do was take that same code and make a few small alterations:
1. Flip the comparison operator from `<` to `>=` in the `.loc` method
2. Switch `.bar` to `.barh`
3. Tweak the size a bit
4. Change the labels

```Python
# Select only outliers and create bar chart.
(fy_2018_expense_medians
 .sort_values("fy_2018_actual_expense", ascending=True)
 .loc[fy_2018_expense_medians.fy_2018_actual_expense >= 40000000, "fy_2018_actual_expense"]
 .plot.barh(
     figsize=(10,4), 
     grid=False, 
     title="Median Expense by Government Agency"
 ))
# Set axis labels.
plt.xlabel("FY18 Median Expense in Tens of Millions")
plt.ylabel("Agency Name")
```

![png]({static}/images/median_top_two.png)

# Conclusion

The `tidyverse` workflow offers a powerful human readable workflow for EDA. Using parenthesis (and backslashes sparingly), we can achieve some of the same flexibility in Python using `Pandas`.