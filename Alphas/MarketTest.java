import cProfile

import time
import yfinance as yf
import csv
import datetime
import strategy
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from helper import (
    series_index_to_dates,
    process_pair,
    check_hold,
    add_to_daytracker,
    remove_from_daytracker,
    get_market_valid_times,
    slice_database_by_dates,
    get_market_end_date,
    remove_stop_loss_from_daytracker,
    check_stop_loss,
)
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from data_download import get_spy_data, get_spy_index_data

# from helper import get_market_start_date

if not os.path.exists(os.path.join("Data", "spy.csv")):
    raise ("ERROR")
    # get_spy_data()

settings = strategy.get_settings()

# Linear list of trades made
trades_made = []

# Current positions organized by ticker
positions = {}

# Tracks days held, updates based on sales, and auto sells at certain date
daytracker = defaultdict(list)

current_capital = settings["INITIAL_CAPITAL"]

day_number = 0

orders_index = []
orders = pd.DataFrame()
database = pd.DataFrame()
database_index = []

portfolio_history = pd.Series()


class Trade:
    def __init__(self, trade_id, symbol, quantity, price, timestamp, type):
        self.trade_id = trade_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.type = type


def calculate_commission(trade):
    if settings.get("C_TYPE") == "PERCENT":
        return trade.price * trade.quantity * settings["C_PERCENT"]
    elif settings.get("C_TYPE") == "FLAT":
        return settings["C_FLAT"]
    elif settings.get("C_TYPE") == "NONE":
        return 0
    else:
        raise TypeError("No valid commision")


# calculates value for positions TO NOW
def portfolio_value(latest=True, date: str = ""):
    total_value = current_capital
    if latest:
        for symbol, position in positions.items():
            stock = yf.Ticker(symbol)
            current_price = stock.history(period="1d")["Close"].iloc[-1]

            # Calculate the current value of the position
            position_value = position["quantity"] * current_price
            total_value += position_value
    else:
        for symbol, position in positions.items():
            current_price = database.loc[date][symbol]

            position_value = position["quantity"] * current_price
            total_value += position_value

    return total_value


def buy_stock(symbol, quantity, price, timestamp: str):  # 2018-06-20
    global current_capital, day_number  # Declare global to update the outer variable
    new_trade = Trade(
        len(trades_made),
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        type="buy",
    )

    if quantity == 0:
        # since we are portion-sizing based on USD, there is a chance we get quantity of zero
        # avoid division by zero
        return

    if current_capital >= (quantity * price + calculate_commission(new_trade)):
        trades_made.append(new_trade)

        # Calculate the total cost of the purchase, including commission
        total_cost = (quantity * price) + calculate_commission(new_trade)

        if symbol in positions:
            positions[symbol]["quantity"] += quantity
            positions[symbol]["avg_price"] = (
                positions[symbol]["avg_price"] * positions[symbol]["quantity"]
                + quantity * price
            ) / (positions[symbol]["quantity"] + quantity)
        else:
            positions[symbol] = {"quantity": quantity, "avg_price": price}

        daytracker = add_to_daytracker(daytracker, quantity, symbol, day_number, price)

        current_capital -= total_cost
    else:
        print("Not enough capital to buy", quantity, "shares of", symbol)


def sell_stock(symbol, quantity, price, timestamp):
    new_trade = Trade(
        len(trades_made),
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        type="sell",
    )
    global current_capital  # Declare global to update the outer variable

    if (
        new_trade.symbol in positions
        and positions[new_trade.symbol]["quantity"] >= quantity
    ):
        # Calculate the total revenue from the sale, after deducting commission
        total_revenue = (quantity * new_trade.price) - calculate_commission(new_trade)

        # Update position
        positions[new_trade.symbol]["quantity"] -= quantity
        # daytracker = remove_from_daytracker(symbol, quantity, daytracker)

        # Check if all shares are sold for this position
        if positions[new_trade.symbol]["quantity"] == 0:
            del positions[new_trade.symbol]

        # Update current capital
        current_capital += total_revenue

    else:
        print("Not enough shares to sell or invalid trade.", new_trade)

def calculate_PnL(portfolio_history):
    return portfolio_history.iloc[-1] - portfolio_history.iloc[0]

def calculate_percentage_returns(portfolio_history):
    return (portfolio_history.iloc[-1] / portfolio_history.iloc[0] - 1) * 100

def calculate_volatility(portfolio_history):
    daily_returns = portfolio_history.pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)  # 252 trading days in a year

def calculate_max_drawdown(portfolio_history):
    # Calculate the cumulative maximum value for each point in time
    cumulative_max = portfolio_history.cummax()
    # Calculate drawdown as the difference between current and cumulative max
    drawdown = (portfolio_history - cumulative_max) / cumulative_max
    return drawdown.min()

def calculate_sharpe_ratio(portfolio_history):
    daily_returns = portfolio_history.pct_change().dropna()
    # Subtract the risk-free rate (assumed to be 0 here) from the returns
    excess_returns = daily_returns - (0 / 252)
    # Calculate the Sharpe Ratio
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_win_rate(trades_made):
    wins = [trade for trade in trades_made if trade.type == 'sell' and trade.price > positions[trade.symbol]["avg_price"]]
    return len(wins) / len(trades_made) if trades_made else 0


# STRUCTURE SHOULD BE [[BUY, symbol, quantity], [SELL, symbol, quantity]]


def run_daily_instructions(current_day: str, instructions=list[list]):
    # print(instructions, type(instructions))
    # Name: 2018-06-20, dtype: object [list([None, 'ADI', 10]) list([None, 'AJG', 10])] <class 'numpy.ndarray'> Index(['A ADI', 'A AJG'], dtype='object') <class 'str'> A AJG

    # [list([None, 'ADI', 10]) list([None, 'AJG', 10])] <class 'numpy.ndarray'>
    # TODO: Add concurrency
    for instruction in instructions:
        for order in instruction:
            order_type, symbol, quantity = order

            print("CURRENT DATE", current_day, type(current_day))

            # current_date_str = current_day.strftime("%Y-%m-%d %H:%M:%S")
            print(instructions)
            print(database.loc[current_day][symbol])
            price = float(database.loc[current_day][symbol])

            if order_type == "BUY":
                buy_stock(symbol, quantity, price, current_day)
            elif order_type == "SELL":
                sell_stock(symbol, quantity, price, current_day)
            else:
                print(f"Invalid order{order}")
                return

            print(order_type, symbol, current_day, price)

            print(f"{order_type: <4} {symbol: <4} on {current_day}: {float(price):.2f}")


def run_timeline(orders: pd.DataFrame, start_date: str, end_date: str):
    """
    Runs all instructions from a dataframe
    """
    format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_date, format)
    end_date = datetime.datetime.strptime(end_date, format)
    to_sell = []
    global day_number, daytracker, portfolio_history

    day_number = 0

    # Set the 'Date' column as the index
    # database.set_index("Date", inplace=True)
    # database.sort_index(inplace=True)

    # TODO: add concurrency
    for current_date in tqdm(orders.index):  # fix
        # Check if the date exists in the index
        # try:
        day_number += 1
        data_rows = orders.loc[current_date]
        instructions = data_rows
        """print(
            instructions,
            instructions.values,
            type(instructions.values),
            instructions.index,
            type(instructions.index[0]),
            instructions.index[-1],
        )"""

        # print("VAlues: ", instructions.values.tolist(), to_sell)
        run_daily_instructions(current_date, instructions.values.tolist())

        # update daytracker. Will move into function if this works

        portfolio_history = save_portfolio_value(portfolio_history, current_date)
        # print(current_date)
        """except Exception as e:
            print(f"No data available for {current_date}, Error: {e}")
            print(current_date, current_date, orders.loc[current_date])
            input()"""

    return None


def plot_all(series: pd.Series, start_date, end_date):
    fig, ax1 = plt.subplots()
    spy_data = get_spy_index_data(
        start_date=start_date, end_date=get_market_end_date(end_date=end_date)
    )
    spy_data_normalized = spy_data.apply(lambda x: (x / (spy_data.iloc[0]) - 1) * 100)
    series_data_normalized = series.apply(lambda x: (x / (series.iloc[0]) - 1) * 100)
    # series_normalized = series.apply(lambda x: x.div(x.iloc[0].subtract(1).mul(100)))

    ax1.plot(database.index, spy_data_normalized.values, label="SPY")

    # df.index = df.index.map(lambda time: time.strftime("%Y-%m-%d"))
    # create df by calling portfolio_value() every day
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Percentage Returns")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title(f"Factor Investing: {start_date} - {end_date}")
    # ax2 = ax1.twinx()
    ax1.plot(
        database.index[-len(series_data_normalized) :],
        series_data_normalized.values,
        label="Factor Investing",
    )
    # ax2.set_ylabel("Percentage Returns")
    # ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    every_nth = 200
    for n, label in enumerate(ax1.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    ax1.legend()
    fig.tight_layout()

    fig.savefig(os.path.join("StatArb", "Backtests", f"{start_date}_{end_date}.png"))

    fig.show()
    print(
        f"Spy: {spy_data.values[-1]}, {spy_data_normalized.values[-1]}%\Factor Investing Model: {series.values[-1]},  {series_data_normalized.values[-1]}"
    )


def save_portfolio_value(series: pd.Series, current_date: str, latest=False):
    # global portfolio_history
    series.at[len(series)] = portfolio_value(latest, current_date)
    return series


# does not account for stock splits
# database currently contains stocks from current s&p 500, if stocks leave/rejoin it gets weird
if __name__ == "__main__":
    portfolio_history = pd.Series()
    start_time = time.time()
    start_date = "2018-06-12"
    end_date = "2022-01-03"

    get_spy_data(
        start_date=start_date, end_date=get_market_end_date(end_date=end_date)
    )  # We need to call market_end_date since yfinance doesn't download the last day
    # https://github.com/ranaroussi/yfinance/issues/1445

    root = "StatArb"
    csv_path = "spy.csv"
    database = pd.read_csv(os.path.join(root, csv_path))
    database_index = list(map(lambda x: x.split(" ")[0], database["Date"].tolist()))
    database["Date"] = database_index
    database.set_index("Date", inplace=True)

    database = slice_database_by_dates(
        database=database, start_date=start_date, end_date=end_date
    )

    # print(database, database.head(), database.tail(), len(database))
    # database.sort_index(inplace=True)

    run_timeline(orders, orders_index[0], orders_index[-1])

    PnL = calculate_PnL(portfolio_history)
    percentage_returns = calculate_percentage_returns(portfolio_history)
    volatility = calculate_volatility(portfolio_history)
    max_drawdown = calculate_max_drawdown(portfolio_history)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_history)
    win_rate = calculate_win_rate(trades_made)

    print(portfolio_history)

    print(f"Current Capital: {float(current_capital):.2f}")
    print(f"Current Portfolio Value: {float(portfolio_value()):.2f}")
    print("Positions:", positions)
    # print(f"Day tracker: ", daytracker)

    time_diff = time.time() - start_time

    print(f"Done. Took {time_diff:.2f} seconds.")

    

    print(f"PnL: {PnL:.2f}")
    print(f"Percentage Returns: {percentage_returns:.2f}%")
    print(f"Volatility: {volatility:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Win Rate: {win_rate:.2%}")

    plot_all(portfolio_history, start_date, end_date)


