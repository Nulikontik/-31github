import pandas as pd
import prophet
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import os
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange
from google.analytics.data_v1beta.types import Dimension
from google.analytics.data_v1beta.types import Metric
from google.analytics.data_v1beta.types import Filter
from google.analytics.data_v1beta.types import FilterExpression
from google.analytics.data_v1beta.types import FilterExpressionList
from google.analytics.data_v1beta.types import RunReportRequest

PROPERTY_ID = '280409742'
START_DATE = '2023-01-01'
END_DATE = '2023-11-01'
PERIODS = 11
FREQ = 'M'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "first-prediction-seo-e9c9b01b2d29.json"


def ga4(PROPERTY_ID, START_DATE, END_DATE):
    client = BetaAnalyticsDataClient()

    request = RunReportRequest(property=f"properties/{PROPERTY_ID}",
                               dimensions=[Dimension(name='date')],
                               metrics=[Metric(name='eventCount')],
                               date_ranges=[DateRange(start_date=START_DATE, end_date=END_DATE)],
                               dimension_filter=FilterExpression(and_group=FilterExpressionList(expressions=[
                                   FilterExpression(filter=Filter(field_name='sessionDefaultChannelGrouping',
                                                                  string_filter=Filter.StringFilter(
                                                                      value='Organic Search',
                                                                      match_type=Filter.StringFilter.MatchType(1)))),
                                   FilterExpression(filter=Filter(field_name='eventName',
                                                                  string_filter=Filter.StringFilter(
                                                                      value='session_start',
                                                                      match_type=Filter.StringFilter.MatchType(1))))])))
    response = client.run_report(request)

    x, y = ([] for i in range(2))
    for row in response.rows:
        x.append(row.dimension_values[0].value)
        y.append(row.metric_values[0].value)
        print(row.dimension_values[0].value, row.metric_values[0].value)

    return x, y


def forecasting(x, y, p, f):
    print('Prophet %s' % prophet.__version__)

    data = {'ds': x, 'y': y}
    df = pd.DataFrame(data, columns=['ds', 'y'])
    m = Prophet(growth='linear',
                changepoint_prior_scale=0.05,
                seasonality_mode='additive',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                holidays=None
                )

    m.fit(df)
    future = m.make_future_dataframe(periods = PERIODS, freq = FREQ)
    forecast = m.predict(future)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    fig = m.plot(forecast, xlabel='Date', ylabel='Visits')
    add_changepoints_to_plot(fig.gca(), m, forecast)
    m.plot_components(forecast)
    plt.show()

if __name__ == "__main__":
    channel_group, event_count = ga4(PROPERTY_ID, START_DATE, END_DATE)
    forecasting(channel_group, event_count, PERIODS, FREQ)