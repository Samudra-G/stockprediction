# cython: language_level=3
cimport cython
import numpy as np  # use np for numpy
cimport numpy as cnp  # if you need to use Cython-specific numpy extensions
from libc.stdlib cimport malloc, free  # Import malloc and free
from libc.string cimport memcpy        # Import memcpy for memory copying
from libcpp.vector cimport vector

cdef extern from "trading.h":
    vector[double] moving_averages(const vector[double]& prices, int window_size)
    double calculate_volatility(const vector[double]& highs, const vector[double]& lows)
    void calcRSI(const vector[double]& closePrices, int period, vector[double]& rsiData)
    double calculate_dividend_yield(const vector[double]& dividends, const vector[double]& closing_prices)


# Wrapping C++ functions into Python-exposed functions
def py_moving_averages(list prices, int window_size):
    cdef vector[double] c_prices
    cdef vector[double] c_averages
    cdef double value
    cdef list py_averages
    for value in prices:
        c_prices.push_back(value)
    c_averages = moving_averages(c_prices, window_size)
    py_averages = [val for val in c_averages]
    return py_averages

def py_calculate_volatility(list highs, list lows):
    cdef vector[double] c_highs
    cdef vector[double] c_lows
    cdef double value
    for value in highs:
        c_highs.push_back(value)
    for value in lows:
        c_lows.push_back(value)
    return calculate_volatility(c_highs, c_lows)

def py_calculate_rsi(list prices, int period):
    cdef vector[double] c_prices
    cdef vector[double] c_rsi
    cdef double value
    cdef list py_rsi
    
    for value in prices:
        c_prices.push_back(value)
    
    calcRSI(c_prices, period, c_rsi)
    
    py_rsi = [val for val in c_rsi]
    return py_rsi

def py_calculate_dividend_yield(double dividend, double closing_price):
    return calculate_dividend_yield([dividend], [closing_price])    
