"""

NOTE:
Any absolute times expressed in seconds (generally assigned to variable 't' throughout the rest of these modules)
are assumed to be expressed in terms of a Julian date converted to total seconds.
I.e., J2000 is a JD of 2451545.0, which would be expressed as 2451545*86400 seconds.

"""

from datetime import datetime

J2000      = 2451545.0  # JD

def str2time(time_str):
    """Return the time as a Julian Date (UTC) in days"""
    date = datetime.strptime(time_str, '%d %b %Y, %H:%M:%S.%f %Z')
    J2K = datetime.strptime('01 Jan 2000, 12:00:00.00 UTC', '%d %b %Y, %H:%M:%S.%f %Z')
    
    # Test case
    # test_date = datetime.strptime('01 Dec 2017, 00:00:48.000383 UTC', '%d %b %Y, %H:%M:%S.%f %Z')
    # assert (test_date - J2000).total_seconds()/86400 + J2000 == 2458088.50055556
    return (date - J2K).total_seconds()/86400 + J2000
