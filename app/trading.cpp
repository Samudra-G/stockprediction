#include <iostream>
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>

std::vector<double> moving_averages(const std::vector<double>& prices, int window_size) {
    std::vector<double> averages;
    if (window_size <= 0 || prices.size() < window_size) return averages;

    double sum = 0.0;
    for (int i = 0; i < window_size; ++i) {
        sum += prices[i];
    }
    averages.push_back(sum / window_size);

    for (size_t i = window_size; i < prices.size(); ++i) {
        sum += prices[i] - prices[i - window_size];
        averages.push_back(sum / window_size);
    }
    return averages;
}

double calculate_volatility(const std::vector<double>& highs, const std::vector<double>& lows) {
    size_t size = highs.size();
    if (size != lows.size() || size == 0) return 0.0;

    double sum_volatility = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_volatility += highs[i] - lows[i];
    }

    return sum_volatility / size;
}

void calcRSI(const std::vector<double>& closePrices, int period, std::vector<double>& rsiData) {
    if (closePrices.size() < period) {
        std::cout << "Not enough data for RSI calculation" << std::endl;
        return;
    }

    std::vector<double> gains(period, 0.0), losses(period, 0.0);
    double avgGain = 0.0, avgLoss = 0.0;

    // Calculate initial average gain and loss
    for (int i = 1; i < period; ++i) {
        double change = closePrices[i] - closePrices[i - 1];
        if (change > 0) gains[i] = change;
        else losses[i] = -change;
    }

    avgGain = std::accumulate(gains.begin(), gains.end(), 0.0) / period;
    avgLoss = std::accumulate(losses.begin(), losses.end(), 0.0) / period;

    // Calculate RSI
    for (size_t i = period; i < closePrices.size(); ++i) {
        double change = closePrices[i] - closePrices[i - 1];
        double gain = change > 0 ? change : 0;
        double loss = change < 0 ? -change : 0;

        avgGain = ((avgGain * (period - 1)) + gain) / period;
        avgLoss = ((avgLoss * (period - 1)) + loss) / period;

        double rs = avgLoss == 0 ? 0 : avgGain / avgLoss;
        double rsi_value = avgLoss == 0 ? 100.0 : 100.0 - (100.0 / (1.0 + rs));
        rsiData.push_back(rsi_value);
    }
}

double calculate_dividend_yield(const double dividend, const double closing_price) {
    if (closing_price != 0) {
        return (dividend / closing_price) * 100.0; 
    }
    return 0.0; 
}

int main() {
    return 0;
}
