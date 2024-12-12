#ifndef TRADING_H
#define TRADING_H

#include <vector>

std::vector<double> moving_averages(const std::vector<double>& prices, int window_size);
double calculate_volatility(const std::vector<double>& highs, const std::vector<double>& lows);
void calcRSI(const std::vector<double>& closePrices, int period, std::vector<double>& rsiData);
double calculate_dividend_yield(const double dividend, const double closing_price);

#endif // TRADING_H
