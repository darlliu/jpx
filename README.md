# jpx
JPX comp related stuff

# ideas

1. GBT + a bunch of descriptive features + prices and moments. Feature selection -> ranking task.

2. embeddings + prices, online train, ranking task.

other stuff probably can't really do (full cross embeddings etc)

# code structure

`data.py` read data through file and API. calculate sharpe ratio etc
- consider standardizing prices later

`mode.py` produces some simple models to try