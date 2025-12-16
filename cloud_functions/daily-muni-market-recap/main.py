'''
'''
# Standard library imports
import io, os
import smtplib
from datetime import datetime

# Email-related imports
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Third-party data handling libraries
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
import seaborn as sns
from pytz import timezone

# Plotting and visualization
import matplotlib.pyplot as plt

# Google Cloud imports
from google.cloud import bigquery, storage, secretmanager

# Local imports
from queries import DAILY_STATS_QUERY

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/tsuyoshikameda/Git/ficc/cloud_functions/daily-muni-market-recap/creds.json'

EASTERN = timezone('US/Eastern')
BQ_CLIENT = bigquery.Client()


# Helper functions
def read_distribution_list():
    """
    Reads distribution list from Google Cloud Storage
    Returns: pandas DataFrame with the distribution list
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()
    
    # Get the bucket and blob
    bucket = client.bucket('ficc-email-lists-prod')
    blob = bucket.blob('dist.csv')
    
    # Download the contents as string
    content = blob.download_as_string()
    
    # Convert to DataFrame
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    
    return df

def sqltodf(sql, bq_client, job_config=None):
    bqr = bq_client.query(sql, job_config=job_config).result()
    return bqr.to_dataframe()

def get_data():
    today = datetime.now(EASTERN).date()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date", "DATE", today),
        ]
    )
    
    new_results = sqltodf(DAILY_STATS_QUERY, BQ_CLIENT, job_config)
    return new_results

def create_institutional_strengtd_graph(df):
    """
    Creates a line plot showing institutional municipal 'strength' (spelled 'strengtd' in the function name 
    to match your original code) indicator.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create line plot
    ax.plot(df['trade_date'], df['customer_sell_trades_over_5_MM'], 
            label='Customer Sold', color='navy', linewidth=2)
    ax.plot(df['trade_date'], df['customer_buy_trades_over_5_MM'], 
            label='Customer Bought', color='forestgreen', linewidth=2)
    
    # Customize plot
    ax.set_title('Institutional Municipal Strength Indicator\n(number of trades > $5 million par)',
                 fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Trades')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_daily_trade_count_graph(df):
    """
    Creates a bar plot of daily trade counts over the last 10 business days.
    """
    df['weekday'] = df['trade_date'].dt.dayofweek
    # 0=Monday through 4=Friday
    df_business = df[df['weekday'].isin([0, 1, 2, 3, 4])]
    df_last_10 = df_business.sort_values('trade_date').tail(10)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    dealer = df_last_10['num_dealer_trades'].values
    buy = df_last_10['num_customer_buy'].values
    sell = df_last_10['num_customer_sell'].values

    ax.bar(range(10), dealer, width=0.7, label='Inter-Dealer', color='royalblue')
    ax.bar(range(10), buy, bottom=dealer, width=0.7, label='Bid Side', color='orange')
    ax.bar(range(10), sell, bottom=dealer + buy, width=0.7, label='Offer Side', color='green')

    for i in range(10):
        ax.text(i, dealer[i] / 2, f"{dealer[i]:,}", ha='center', va='center', color='white', fontsize=9)
        ax.text(i, dealer[i] + buy[i] / 2, f"{buy[i]:,}", ha='center', va='center', color='white', fontsize=9)
        ax.text(i, dealer[i] + buy[i] + sell[i] / 2, f"{sell[i]:,}", ha='center', va='center', color='white', fontsize=9)

    dates = df_last_10['trade_date'].dt.strftime('%m/%d/%y')
    plt.xticks(range(10), dates, rotation=45)
    
    # Customize plot
    ax.set_title('Daily Trade Count Past 10 Days', fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('# of Trades')
    
    # Format y-axis with comma separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Set y-axis limits
    maxLimit = df_last_10['trade_count'].max() * 1.1
    ax.set_ylim(0, maxLimit)
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # key
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title='Trade Type')
    plt.tight_layout()
    return fig

def create_average_par_traded_graph(df):
    """
    Creates a horizontal bar plot of average par traded for the last 10 business days.
    """
    df['weekday'] = df['trade_date'].dt.dayofweek
    # 0=Monday through 4=Friday
    df_business = df[df['weekday'].isin([0, 1, 2, 3, 4])]
    df_last_10 = df_business.sort_values('trade_date').tail(10)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    bars = ax.barh(range(10), df_last_10['average_par_traded'], 
                   color='royalblue', height=0.7)
    
    # Add date labels on y-axis
    dates = df_last_10['trade_date'].dt.strftime('%m/%d/%y')
    plt.yticks(range(10), dates)
    
    # Customize plot
    ax.set_title('Average Par Traded Past 10 Days', fontsize=12, pad=15)
    ax.set_xlabel('Average Par Traded ($)')
    
    # Format x-axis with dollar signs and commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add gridlines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_trade_volume_graph(df):
    """
    Creates a bar plot for total trade volume (in millions) by day for the last 10
    business days. The y-axis is labeled in millions of dollars with a '$' sign.
    """
    # Filter the DataFrame to only include business days
    df['weekday'] = df['trade_date'].dt.dayofweek
    # 0=Monday, ..., 4=Friday
    df_business = df[df['weekday'].isin([0, 1, 2, 3, 4])]
    
    # Sort and pick the last 10 business days
    df_last_10 = df_business.sort_values('trade_date').tail(10)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the volume in millions
    ax.bar(range(10), df_last_10['total_trade_volume'] / 1_000_000,
           color='royalblue', width=0.7)
    
    # Set x-axis labels
    dates = df_last_10['trade_date'].dt.strftime('%m/%d/%y')
    plt.xticks(range(10), dates, rotation=45)
    
    # Set titles and labels
    ax.set_title('Total Trade Volume by Day (Past 10 Days)', fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Volume (in millions)')
    
    # Format y-axis with a dollar sign
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def create_yield_5yr_graph(df):
    """
    Creates a line plot of average daily yields for 3%, 4%, and 5% coupons with 5-year maturities.
    """
    df_last_30 = df.sort_values('trade_date').tail(30)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_3_percent_coupon_5_yr'],
            label='3% Coupon', color='blue')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_4_percent_coupon_5_yr'],
            label='4% Coupon', color='green')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_5_percent_coupon_5_yr'],
            label='5% Coupon', color='orange')
    
    ax.set_title('Average Daily Yield for Investment Grade Trades (5 Year)', fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield (%)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_yield_10yr_graph(df):
    """
    Creates a line plot of average daily yields for 3%, 4%, and 5% coupons with 10-year maturities.
    """
    df_last_30 = df.sort_values('trade_date').tail(30)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_3_percent_coupon_10_yr'],
            label='3%', color='blue')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_4_percent_coupon_10_yr'],
            label='4%', color='green')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_5_percent_coupon_10_yr'],
            label='5%', color='orange')
    
    ax.set_title('Average Daily Yield for Investment Grade Trades (10 Year)', fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield (%)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_yield_20yr_graph(df):
    """
    Creates a line plot of average daily yields for 3%, 4%, and 5% coupons with 20-year maturities.
    """
    df_last_30 = df.sort_values('trade_date').tail(30)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_3_percent_coupon_20_yr'],
            label='3%', color='blue')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_4_percent_coupon_20_yr'],
            label='4%', color='green')
    ax.plot(df_last_30['trade_date'], df_last_30['avg_yield_for_5_percent_coupon_20_yr'],
            label='5%', color='orange')
    
    ax.set_title('Average Daily Yield for Investment Grade Trades (20 Year)', fontsize=12, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield (%)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_html_report(df, row):
    """
    Generates the HTML for the email and returns both the HTML string 
    and the list of image buffers to be embedded.
    """
    # Create graphs and save to buffers
    strength_indicator_fig = create_institutional_strengtd_graph(df)
    strength_indicator_buf = io.BytesIO()
    strength_indicator_fig.savefig(strength_indicator_buf, format='png', dpi=300, bbox_inches='tight')
    strength_indicator_buf.seek(0)
    plt.close(strength_indicator_fig)
    
    trade_count_fig = create_daily_trade_count_graph(df)
    trade_count_buf = io.BytesIO()
    trade_count_fig.savefig(trade_count_buf, format='png', dpi=300, bbox_inches='tight')
    trade_count_buf.seek(0)
    plt.close(trade_count_fig)

    avg_par_fig = create_average_par_traded_graph(df)
    avg_par_buf = io.BytesIO()
    avg_par_fig.savefig(avg_par_buf, format='png', dpi=300, bbox_inches='tight')
    avg_par_buf.seek(0)
    plt.close(avg_par_fig)

    trade_volume_fig = create_trade_volume_graph(df)
    trade_volume_buf = io.BytesIO()
    trade_volume_fig.savefig(trade_volume_buf, format='png', dpi=300, bbox_inches='tight')
    trade_volume_buf.seek(0)
    plt.close(trade_volume_fig)
    
    yield_5yr_fig = create_yield_5yr_graph(df)
    yield_5yr_buf = io.BytesIO()
    yield_5yr_fig.savefig(yield_5yr_buf, format='png', dpi=300, bbox_inches='tight')
    yield_5yr_buf.seek(0)
    plt.close(yield_5yr_fig)
    
    yield_10yr_fig = create_yield_10yr_graph(df)
    yield_10yr_buf = io.BytesIO()
    yield_10yr_fig.savefig(yield_10yr_buf, format='png', dpi=300, bbox_inches='tight')
    yield_10yr_buf.seek(0)
    plt.close(yield_10yr_fig)
    
    yield_20yr_fig = create_yield_20yr_graph(df)
    yield_20yr_buf = io.BytesIO()
    yield_20yr_fig.savefig(yield_20yr_buf, format='png', dpi=300, bbox_inches='tight')
    yield_20yr_buf.seek(0)
    plt.close(yield_20yr_fig)

    # Styled div acting as a logo
    logo_html = """
        <table width="80" height="80" style="background:#1a1a1a; border-collapse: collapse; border-spacing: 0;">
            <tr>
                <td align="center" valign="middle">
                    <span style="color:#00BFFF; font-size:16px; font-family:Arial, sans-serif; font-weight:bold; text-decoration:none;">ficc . ai</span>
                </td>
            </tr>
        </table>
    """

    message = f"""
    <html>
    <head>
        <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            display: grid;
            grid-template-columns: 40px 1fr;
            gap: 20px;
            align-items: center;
            margin-bottom: 30px;
        }}
        .stats-table {{
            width: 100%;
            margin: 20px 0;
            border-collapse: separate;
            border-spacing: 20px 0;
            font-size:18px;
        }}
        .stats-table td {{
            vertical-align: top;
            width: 50%;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table td, .data-table th {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        .data-table th {{
            background-color: #f5f5f5;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .data-table tr:nth-child(odd) {{
            background-color: #ffffff;
        }}
        </style>
    </head>
    <body>
        <div class="header">
            {logo_html}
            <div>
                <h1 style="margin: 0;">Daily Municipal Market Recap</h1>
                <p style="margin: 5px 0;">{datetime.now().strftime("%B %d, %Y")}</p>
            </div>
        </div>

        <table class="stats-table">
            <tr>
                <td>
                    <div style="margin: 8px 0;">• New Issue With Most Trades Today: {row['most_actively_traded_new_issue']}<br>{row['description']}</div>
                    <div style="margin: 8px 0;">• Number of trades today: {row['trade_count']:,}</div>
                    <div style="margin: 8px 0;">• Average par traded: ${row['average_par_traded']:,.0f}</div>
                    <div style="margin: 8px 0;">• Total volume of trades today: ${row['total_trade_volume']:,.0f}</div>
                </td>
                <td>
                    <div style="margin: 8px 0;">• Seasoned Issue With Most Trades Today: {row['most_actively_traded_seasoned_cusip']}<br>{row['description_2']}</div>
                    <div style="margin: 8px 0;">• Number of dealer-to-dealer trades: {row['num_dealer_trades']:,}</div>
                    <div style="margin: 8px 0;">• Number of customer bought trades: {row['num_customer_buy']:,}</div>
                    <div style="margin: 8px 0;">• Number of customer sold trades: {row['num_customer_sell']:,}</div>
                </td>
            </tr>
            <tr>
                <td>
                    <div style="margin: 8px 0;">• Number of customer sold trades above $5MM: {row['customer_sell_trades_over_5_MM']:,}</div>
                    <div style="margin: 8px 0;">• Number of dealer-to-dealer trades above $5MM: {row['dealer_trades_over_5_MM']:,}</div>
                    <div style="margin: 8px 0;">• Number of customer bought trades above $5MM: {row['customer_buy_trades_over_5_MM']:,}</div>
                </td>
            </tr>
        </table>

        <h2>Figure 1: Average Price & Yield of Investment Grade Trades</h2>
        <table class="data-table">
            <tr>
                <th>Maturity</th>
                <th colspan="2">3% Coupon</th>
                <th colspan="2">4% Coupon</th>
                <th colspan="2">5% Coupon</th>
            </tr>
            <tr>
                <th>Average</th>
                <th>Price</th>
                <th>Yield</th>
                <th>Price</th>
                <th>Yield</th>
                <th>Price</th>
                <th>Yield</th>
            </tr>
            <tr>
                <td>5 year</td>
                <td>${row['avg_price_for_3_percent_coupon_5_yr']:.3f}</td>
                <td>{row['avg_yield_for_3_percent_coupon_5_yr']:.2f}%</td>
                <td>${row['avg_price_for_4_percent_coupon_5_yr']:.3f}</td>
                <td>{row['avg_yield_for_4_percent_coupon_5_yr']:.2f}%</td>
                <td>${row['avg_price_for_5_percent_coupon_5_yr']:.3f}</td>
                <td>{row['avg_yield_for_5_percent_coupon_5_yr']:.2f}%</td>
            </tr>
            <tr>
                <td>10 year</td>
                <td>${row['avg_price_for_3_percent_coupon_10_yr']:.3f}</td>
                <td>{row['avg_yield_for_3_percent_coupon_10_yr']:.2f}%</td>
                <td>${row['avg_price_for_4_percent_coupon_10_yr']:.3f}</td>
                <td>{row['avg_yield_for_4_percent_coupon_10_yr']:.2f}%</td>
                <td>${row['avg_price_for_5_percent_coupon_10_yr']:.3f}</td>
                <td>{row['avg_yield_for_5_percent_coupon_10_yr']:.2f}%</td>
            </tr>
            <tr>
                <td>20 year</td>
                <td>${row['avg_price_for_3_percent_coupon_20_yr']:.3f}</td>
                <td>{row['avg_yield_for_3_percent_coupon_20_yr']:.2f}%</td>
                <td>${row['avg_price_for_4_percent_coupon_20_yr']:.3f}</td>
                <td>{row['avg_yield_for_4_percent_coupon_20_yr']:.2f}%</td>
                <td>${row['avg_price_for_5_percent_coupon_20_yr']:.3f}</td>
                <td>{row['avg_yield_for_5_percent_coupon_20_yr']:.2f}%</td>
            </tr>
        </table>
        <p style="font-size: 12px;">Source: ficc.ai, Municipal Securities Rulemaking Board</p>
           
        <h2>Figure 2: Institutional Municipal Strength Indicator</h2>
        <img src="cid:image1" style="width:850px; height:500px">
        
        <h2>Figure 3: Daily Trade Count Past 10 Days</h2>
        <img src="cid:image2" style="width:850px; height:500px">
        
        <h2>Figure 4: Average Par Traded Past 10 Days</h2>
        <img src="cid:image3" style="width:850px; height:500px">
        
        <h2>Figure 5: Total Trade Volume by Day Past 10 Days</h2>
        <img src="cid:image4" style="width:850px; height:500px">
        
        <h2>Figure 6: Average Daily Yield for Investment Grade Trades with 5 Year Maturity</h2>
        <img src="cid:image5" style="width:850px; height:500px">
        
        <h2>Figure 7: Average Daily Yield for Investment Grade Trades with 10 Year Maturity</h2>
        <img src="cid:image6" style="width:850px; height:500px">
        
        <h2>Figure 8: Average Daily Yield for Investment Grade Trades with 20 Year Maturity</h2>
        <img src="cid:image7" style="width:850px; height:500px">
        
    </body>
    </html>
    """
    return message, [
        strength_indicator_buf, 
        trade_count_buf, 
        avg_par_buf, 
        trade_volume_buf, 
        yield_5yr_buf, 
        yield_10yr_buf, 
        yield_20yr_buf
    ]


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_email(subject, html_content, images, df_emails, testing=False):
    """
    Sends the email using SMTP with embedded images.
    
    - df_emails: DataFrame containing at least two columns: 'Name' and 'Email'
    - subject: Email subject line
    - html_content: The HTML body of the email
    - images: A list of in-memory image buffers (BytesIO) to be attached inline
    - testing: If True, only BCC gil@ficc.ai. Otherwise, BCC all emails in df_emails.
    """

    # Hard-coded TO address:
    to_address = 'myles@ficc.ai'
    
    # BCC: either just Gil for testing or everyone in df_emails
    if testing:
        bcc_list = ['gil@ficc.ai']  # Only Gil when testing
    else:
        bcc_list = df_emails['Email'].tolist()

    # Sender credentials
    sender_email = access_secret_version('notifications_username')
    password = access_secret_version('notifications_password')
    
    # Build the email message
    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_address
    msg['Bcc'] = ', '.join(bcc_list)  # BCC as comma-delimited string
    
    # Create the alternative section for HTML
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)
    
    # Attach the HTML content
    msg_text = MIMEText(html_content, 'html')
    msg_alternative.attach(msg_text)
    
    # Attach images (inline) with content IDs
    for i, img_buf in enumerate(images, 1):
        img = MIMEImage(img_buf.getvalue())
        img.add_header('Content-ID', f'<image{i}>')
        msg.attach(img)
    
    # Send via Gmail SMTP
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)


class USHolidayCalendarWithGoodFriday(USFederalHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]

def today_is_a_holiday() -> bool:
    '''Determine whether today is a US national holiday.'''
    now = datetime.now(EASTERN)
    today = pd.Timestamp(now).tz_localize(None).normalize()    # `.tz_localize(None)` is to remove the time zone; `.normalize()` is used to remove the time component from the timestamp
    current_year = now.year
    holidays_in_last_year_and_next_year = set(USHolidayCalendarWithGoodFriday().holidays(start=f'{current_year - 1}-01-01',end=f'{current_year + 1}-12-31'))
    if today in holidays_in_last_year_and_next_year:
        print(f'Today, {today}, is a national holiday')
        return True
    return False


def main(args):
    """
    Main function to:
    1) Query BigQuery data
    2) Generate plots
    3) Embed plots into an HTML email
    4) Send that email
    """
    if today_is_a_holiday(): return 'SUCCESS'

    df = get_data()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    today = datetime.now(EASTERN).date()
    today_data = df[df['trade_date'].dt.date == today]
    
    if len(today_data) == 0:
        raise ValueError(f"No data found for today ({today})")
    
    row = today_data.iloc[-1]
    
    # Generate HTML and images
    html_content, images = generate_html_report(df, row)

    # Send email
    recipients = read_distribution_list() #['gil@ficc.ai', 'myles@ficc.ai']
    send_email(
        "ficc.ai Daily Municipal Market Recap", 
        html_content, 
        images, 
        recipients,
        False
    )
    return "Success"

# if __name__ == '__main__':
#     main(None) # for testing purposes
