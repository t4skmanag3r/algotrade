import pandas as pd
import numpy as np


def __get_result_sell(
    ticker, strategy, start_date="2020-01-01"
):  # Function used to detect if stock generated selling signal after buying first
    from algotrade.testing import TestStrategy

    test = TestStrategy(ticker, start_date, strategy)
    if len(test.sell_dates) != 0:
        if test.sell_dates[-1].to_pydatetime() > test.buy_dates[-1].to_pydatetime():
            buy_date = test.buy_dates[-1]
            sell_date = test.sell_dates[-1]
            df = test.df
            buy_price = df.iloc[df.index == buy_date].adjclose.values[0]
            sell_price = df.iloc[df.index == sell_date].adjclose.values[0]
            profit = str(round(sell_price - buy_price, 2)) + "$"
            profit_perc = (
                str(round(((sell_price - buy_price) / buy_price) * 100, 2)) + "%"
            )
            return [
                ticker,
                buy_date,
                str(round(buy_price, 2)) + "$",
                sell_date,
                str(round(sell_price, 2)) + "$",
                profit,
                profit_perc,
            ]
    return None


def getAlerts(tickers, strategy):
    """
    Checks if any stock generated a selling signal after buying first

    Args:
        tickers : list(str)
            list of tickers to watch for selling signals
        strategy : <NewStrategy>
            class strategies.NewStrategy() with set arguments to apply for getting buying, selling signals

    Returns:
        list(list)
            list of lists containing generated sell signals for tickers - [ticker, buy_date, buy_price, sell_date, sell_price, profit_perc]
    """
    return list(
        filter(
            lambda x: x != None,
            [__get_result_sell(ticker, strategy) for ticker in tickers],
        )
    )


# Change message text to html
def sendAlertMail(
    alerts, sender_email_adress=None, email_pass=None, receiver_email=None
):
    """
    Sends email for each ticker alert

    Args:
        alerts : list(list)
            list of alerts generated from alerts.getAlerts() to send
        sender_email_adress : str
            email adress which mail is sent from
        email_pass : str
            password for senders email
        receiver email : str
            email adress which mail is sent to, in other words received, leave empty if same as sender
    """
    if sender_email_adress != None and email_pass != None:
        import smtplib, ssl

        if receiver_email == None:
            receiver_email = sender_email_adress

        port = 465

        context = ssl.create_default_context()

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                server.login(sender_email_adress, email_pass)
                for alert in alerts:
                    message = f"""\
                    Subject: Stock Alert!!!

                    Your strategy generated a sell Alert for {alert[0]}

                    Buy date: {alert[1]}
                    Sell date: {alert[3]}
                    Buy price: {alert[2]}
                    Sell price: {alert[4]}
                    Profit: {alert[5]}
                    Profit percentage: {alert[6]}
                    """
                    server.sendmail(sender_email_adress, receiver_email, message)
        except Exception as e:
            raise e


def playAlertSound(sound_file_path=None):
    """
    Plays sound when called

    Args:
        sound_file_path : str
            directory path to sound file

    """
    from playsound import playsound

    if sound_file_path != None:
        playsound(sound_file_path)


def createAlertSystem(
    tickers,
    strategy,
    send_email=False,
    sound_alert=False,
    sound_alert_path=None,
    sender_email_address=None,
    email_pass=None,
    receiver_email=None,
):
    """
    Initiates the alert system
    Alert system is used to send email and alert the user if stock has generated a sell signal after buying with given strategy

    Args:
        tickers : list(str)
            list of tickers to watch for selling signals
        strategy : <NewStrategy>
            class strategies.NewStrategy() with set arguments to apply for getting buying, selling signals
        send_email : bool
            sends email with alert if True
        sound_alert : bool
            triggers sound alert if True
        sound_alert_path : str
            directory path to sound file
        sender_email_adress : str
            email adress which mail is sent from
        email_pass : str
            password for senders email
        receiver email : str
            email adress which mail is sent to, in other words received, leave empty if same as sender
    """
    alerts = getAlerts(tickers, strategy)

    if send_email and len(alerts) != 0:
        sendAlertMail(alerts, sender_email_address, email_pass, receiver_email)
    if sound_alert and len(alerts) != 0:
        playAlertSound(sound_alert_path)
    return alerts


# def main():
#     from testing import NewStrategy

#     buy_strategies = ["buy_conv", "buy_cloud", "buy_leadspan"]
#     buy_weights = [1, 4, 3]
#     sell_strategies = ["sell_conv", "sell_cloud", "sell_leadspan"]
#     sell_weights = [1, 3, 1]
#     buy_threshold = 4
#     sell_threshold = 3
#     strategy = NewStrategy(buy_strategies, buy_weights, sell_strategies, sell_weights, buy_threshold, sell_threshold)

#     tickers = ['VLO', 'AMD', 'AMAT']
#     sell = createAlertSystem(tickers, strategy,
#         sound_alert=True,
#         sound_alert_path='C:\\Users\\Edvinas\\Downloads\\alert.wav',
#         send_email=True,
#         sender_email_address='',
#         email_pass='',
#         receiver_email=''
#         )
#     for line in sell:
#         print(line)

# if __name__ == '__main__':
#     main()
