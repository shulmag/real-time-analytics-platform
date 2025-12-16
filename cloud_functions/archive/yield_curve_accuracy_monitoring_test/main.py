# @Date:   2023-03-01 13:00:00
'''Compare the difference between our yield curve and MMD's yield curve.'''
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ficc_ycl import *
import pytz

window = 14
threshold_multiple = 3
MMD_maturities = ['1', '5', '10', '15', '30']
t = np.array([1, 5, 10, 15, 30])
est = pytz.timezone('US/Eastern')
fmt = '%Y-%m-%d %H:%M'


def send_error_email(subject, error_message):
    receiver_email = "ficcteam@ficc.ai"
    sender_email = "notifications@ficc.ai"

    recipients = [receiver_email]
    emaillist = [elem.strip().split(',') for elem in recipients]
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    html = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(
        error_message
    )

    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    smtp_server = "smtp.gmail.com"
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, 'ztwbwrzdqsucetbg')
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


def main(args):
    # Load MMD AAA data and get most recent entry
    MMD_data, MMD_last_modified_time = load_mmd_data(last_modified=True)
    MMD_data = MMD_data.pivot_table(
        index='date', columns='maturity', values='AAA'
    ).sort_index()
    MMD_data.columns = MMD_data.columns.astype('str')
    MMD_data = MMD_data * 100  # convert to basis points
    target_MMD_data = MMD_data.iloc[-1, :]  # most recent data
    day_before_target_MMD_data = MMD_data.iloc[
        -2, :
    ]  # data from the day before the most recent day

    target_date = target_MMD_data.name.strftime('%Y-%m-%d')
    day_before_target_date = day_before_target_MMD_data.name.strftime('%Y-%m-%d')
    MMD_last_modified_time = MMD_last_modified_time.astimezone(est).strftime(fmt)

    # Load daily Nelson-Siegel coefficients
    curve_scaler = load_scaler_daily_bq()
    curve_coefficients = load_nelson_siegel_daily_bq()
    shape_params = load_shape_parameter()
    yield_curve_params = curve_scaler.join(curve_coefficients).join(shape_params)
    yield_curve_params.index = pd.to_datetime(yield_curve_params.index)
    yield_curve_params.dropna(inplace=True)
    del curve_scaler, curve_coefficients, shape_params

    # Restrict dataframe to relevant timeframe to avoid redundant computations
    yield_curve_params = yield_curve_params.loc[:target_date].iloc[-window:, :]

    # Calculate daily Nelson-Siegel ycl for relevant timeframe
    ficc_ycl = yield_curve_params.apply(
        lambda x: predict_ytw(
            t,
            x['const'],
            x['exponential'],
            x['laguerre'],
            x['exponential_mean'],
            x['exponential_std'],
            x['laguerre_mean'],
            x['laguerre_std'],
            x['L'],
        ),
        result_type='expand',
        axis=1,
    )
    ficc_ycl.columns = MMD_maturities
    del yield_curve_params

    # Calculate deltas, mean delta and standard deviation
    MMD_ficc_deltas = ficc_ycl - MMD_data.loc[ficc_ycl.index]
    mean_delta = MMD_ficc_deltas.mean()
    std_delta = MMD_ficc_deltas.std()
    upper_bound = mean_delta + threshold_multiple * std_delta
    lower_bound = np.maximum(
        mean_delta - threshold_multiple * std_delta, 0
    )  # Lower bound should be non-negative, it does not make sense if our delta is negative because the spread should be positive between ficc and MMD AAA

    condition1 = MMD_ficc_deltas.iloc[-1, :] >= upper_bound
    condition2 = MMD_ficc_deltas.iloc[-1, :] <= lower_bound
    condition = condition1 + condition2

    email_df = pd.DataFrame()

    # If any deltas exceed upper or lower bound, send alert
    if condition.any():
        message = f'''
        <p>These yield curve differences are over {threshold_multiple} standard deviations away from the {window} day mean difference: maturities {', '.join(condition[condition].index) }. </p> 
        <p>Ficc yield curve and MMD AAA values are effective as of {target_date} end of day. MMD data was scraped on {MMD_last_modified_time} EST.</p>
        <p></p>
        '''

        email_df['Maturity'] = condition1.index
        email_df['Ficc YCL'] = ficc_ycl.loc[target_date].values
        email_df['MMD AAA YCL'] = target_MMD_data.values
        email_df['Difference (Ficc-MMD)'] = MMD_ficc_deltas.loc[target_date].values
        email_df['Day Before Difference'] = MMD_ficc_deltas.loc[
            day_before_target_date
        ].values
        email_df[f'{window} Day Average Difference'] = mean_delta.values
        flag = True

        message += f'<p>{np.round(email_df,2).to_html(index=False, col_space=35)}</p>'

        send_error_email('WARNING! Yield Curve Delta Exceeded Thresholds', message)
        return "ERROR"
    else:
        return "Success"
