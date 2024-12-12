import financial_functions

def test_moving_averages():
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    window_size = 3
    result = financial_functions.py_moving_averages(prices, window_size)
    print(f"Moving Averages: {result}")

def test_calculate_volatility():
    highs = [1.5, 2.5, 3.5, 4.5, 5.5]
    lows = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = financial_functions.py_calculate_volatility(highs, lows)
    print(f"Volatility: {result:.3f}")

def test_calculate_rsi():
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    period = 3
    result = financial_functions.py_calculate_rsi(prices, period)
    print(f"RSI: {result}")

def test_calculate_macd():
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    short_window = 3
    long_window = 5
    signal_window = 2
    result = financial_functions.py_calculate_macd(prices, short_window, long_window, signal_window)
    print(f"MACD: {result}")

def test_calculate_dividend_yield():
    dividends = [0.5, 0.6, 0.7]
    closing_prices = [10.0, 11.0, 12.0]
    result = financial_functions.py_calculate_dividend_yield(dividends, closing_prices)
    print(f"Dividend Yield: {result:.3f}")

if __name__ == "__main__":
    test_moving_averages()
    test_calculate_volatility()
    test_calculate_rsi()
    test_calculate_macd()
    test_calculate_dividend_yield()
