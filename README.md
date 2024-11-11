# TradeNotifier

TradeNotifier is an automated script designed to notify users when specified financial indicators meet predefined criteria. It is particularly useful for monitoring currency exchange rates and triggering alerts based on technical analysis indicators, such as the Relative Strength Index (RSI).

## Features

- Fetches daily foreign exchange data using the Alpha Vantage API.
- Converts raw data into a structured pandas DataFrame.
- Calculates the RSI to determine buy, sell, or hold signals.
- Notifies users when specific indicator criteria are met.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/htseng2/TradeNotifier.git
   cd TradeNotifier
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure API Key**:

   - Replace `YOUR_ALPHA_VANTAGE_API_KEY` in `trade_notifier.py` with your actual Alpha Vantage API key.

2. **Run the Script**:

   ```bash
   python trade_notifier.py
   ```

3. **Receive Notifications**:
   - The script will output buy, sell, or hold signals based on the RSI calculation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).
