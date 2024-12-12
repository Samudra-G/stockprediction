double calculate_dividend_yield(const double dividend, const double closing_price) {
    if (closing_price != 0) {
        return (dividend / closing_price) * 100.0; 
    }
    return 0.0; 
}