"""
TAMSAT download and extraction API source code.

Author: TAMSAT (tamsat@reading.ac.uk)
"""


import os
import wget
import numpy as np
import xarray as xr
import itertools
import pandas as pd
import calendar
from datetime import datetime as dt
from datetime import timedelta as td


def date_df(timestep, startdate, enddate, endmonth):
    """Create dataframe of dates and related information for RFE file creation.

    Parameters
    ----------
    timestep : str
        Timestep - choices are 'daily','pentadal', 'dekadal', 'monthly' and 'seasonal'.
    startdate : str
        Start date (e.g. '2020-01-01').
    enddate : str
        End date (e.g. '2020-01-10').
    endmonth : bool
        If True, changes 'enddate' to the last day of the month.

    Returns
    -------
    dataframe object
        Pandas dataframe with all relevent date information for given timestep.

    """
    # Convert enddate to last day in month
    if endmonth == True:
        enddate_check = pd.to_datetime(enddate, format="%Y-%m-%d")
        if enddate_check.is_month_end == False:
            enddate = (
                pd.to_datetime(enddate, format="%Y-%m-%d")
                + pd.tseries.offsets.MonthEnd()
            )
            enddate = enddate.strftime("%Y-%m-%d")
    
    # Convert start and end date
    start = dt.strptime(startdate, "%Y-%m-%d").date()
    end = dt.strptime(enddate, "%Y-%m-%d").date()
    
    # Create dataframe of dates
    df = pd.date_range(start=start, end=end, name="Date")
    df = df.to_frame(index=False)
    df["Year"] = df.Date.apply(lambda x: x.strftime("%Y"))
    df["Month"] = df.Date.apply(lambda x: x.strftime("%m"))
    df["Day"] = df.Date.apply(lambda x: x.strftime("%d"))
    df["DoY"] = df["Date"].dt.dayofyear
    df["Pentad"] = "0"
    df.loc[df.Day.apply(lambda x: x in ["01", "02", "03", "04", "05"]), "Pentad"] = "1"
    df.loc[df.Day.apply(lambda x: x in ["06", "07", "08", "09", "10"]), "Pentad"] = "2"
    df.loc[df.Day.apply(lambda x: x in ["11", "12", "13", "14", "15"]), "Pentad"] = "3"
    df.loc[df.Day.apply(lambda x: x in ["16", "17", "18", "19", "20"]), "Pentad"] = "4"
    df.loc[df.Day.apply(lambda x: x in ["21", "22", "23", "24", "25"]), "Pentad"] = "5"
    df.loc[
        df.Day.apply(lambda x: x in ["26", "27", "28", "29", "30", "31"]), "Pentad"
    ] = "6"
    df["Dekad"] = "0"
    df.loc[
        df.Day.apply(
            lambda x: x in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        ),
        "Dekad",
    ] = "1"
    df.loc[
        df.Day.apply(
            lambda x: x in ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        ),
        "Dekad",
    ] = "2"
    df.loc[
        df.Day.apply(
            lambda x: x
            in ["21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
        ),
        "Dekad",
    ] = "3"
    df["Season"] = "0"
    df.loc[df.Month.apply(lambda x: x in ["12", "01", "02"]), "Season"] = "DJF"
    df.loc[df.Month.apply(lambda x: x in ["03", "04", "05"]), "Season"] = "MAM"
    df.loc[df.Month.apply(lambda x: x in ["06", "07", "08"]), "Season"] = "JJA"
    df.loc[df.Month.apply(lambda x: x in ["09", "10", "11"]), "Season"] = "SON"
    
    # Now add additional information based on timestep
    if timestep == "daily":
        df["StartDate"] = df.Date.apply(lambda x: x.replace(day=1))
        df["EndDate"] = df.Date
    elif timestep == "pentadal":
        pentad_length = [len(list(g[1])) for g in itertools.groupby(df.Pentad)]
        pentad_cumsum = np.cumsum(pentad_length)
        df = df.iloc[pentad_cumsum - pentad_length]
        df["PentadLength"] = pentad_length
        df["StartDate"] = df.Date.apply(lambda x: x.replace(day=1))
        df["EndDate"] = df.Date + pd.to_timedelta(df.PentadLength - 1, unit="D")
    elif timestep == "dekadal":
        dekad_length = [len(list(g[1])) for g in itertools.groupby(df.Dekad)]
        dekad_cumsum = np.cumsum(dekad_length)
        df = df.iloc[dekad_cumsum - dekad_length]
        df["DekadLength"] = dekad_length
        df["StartDate"] = df.Date.apply(lambda x: x.replace(day=1))
        df["EndDate"] = df.Date + pd.to_timedelta(df.DekadLength - 1, unit="D")
    elif timestep == "monthly":
        month_length = [len(list(g[1])) for g in itertools.groupby(df.Month)]
        month_cumsum = np.cumsum(month_length)
        df = df.iloc[month_cumsum - month_length]
        df["MonthLength"] = month_length
        df["StartDate"] = df.Date.apply(lambda x: x.replace(day=1))
        df["EndDate"] = df.Date + pd.to_timedelta(df.MonthLength - 1, unit="D")
    elif timestep == "seasonal":
        season_length = [len(list(g[1])) for g in itertools.groupby(df.Season)]
        season_cumsum = np.cumsum(season_length)
        df = df.iloc[season_cumsum - season_length]
        df["SeasonLength"] = season_length
        df["StartDate"] = df.Date.apply(lambda x: x.replace(day=1))
        df["EndDate"] = df.Date + pd.to_timedelta(df.SeasonLength - 1, unit="D")
        length = []
        for index, row in df.iterrows():
            if row.Season == "MAM":
                length.append(92)
            elif row.Season == "JJA":
                length.append(92)
            elif row.Season == "SON":
                length.append(91)
            elif row.Season == "DJF" and calendar.isleap(
                int(row.EndDate.to_pydatetime().year)
            ):
                length.append(91)
            elif (
                row.Season == "DJF"
                and calendar.isleap(int(row.EndDate.to_pydatetime().year)) == False
            ):
                length.append(90)
        
        df = df.loc[df.SeasonLength >= length]
    else:
        print("Timestep not recognised")
    
    # Remove last row if timestep isn't complete
    if timestep == "pentadal" or timestep == "dekadal" or timestep == "monthly":
        if check_end(timestep, end) == False:
            df.drop(df.tail(1).index, inplace=True)
    
    return df


def check_end(timestep, date):
    """Check if date is end of TAMSAT standard time periods.

    Parameters
    ----------
    timestep : str
        Timestep - choices are 'pentadal', 'dekadal', 'monthly' and 'seasonal'
    date : datetime
        Datetime object (e.g. 'datetime.date(2020, 1, 6)').

    Returns
    -------
    bool
        Is the current date the end day of the time period?

    """
    if date.month in [1, 3, 5, 7, 8, 10, 12]:
        if timestep == "pentadal":
            if date.day in [5, 10, 15, 20, 25, 31]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "dekadal":
            if date.day in [10, 20, 31]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "monthly":
            if date.day in [31]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "seasonal":
            if date.month in [5, 8] and date.day in [31]:
                timestep_end = True
            else:
                timestep_end = False
    elif date.month in [4, 6, 9, 11]:
        if timestep == "pentadal":
            if date.day in [5, 10, 15, 20, 25, 30]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "dekadal":
            if date.day in [10, 20, 30]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "monthly":
            if date.day in [30]:
                timestep_end = True
            else:
                timestep_end = False
        elif timestep == "seasonal":
            if date.month in [11] and date.day in [30]:
                timestep_end = True
            else:
                timestep_end = False
    elif date.month in [2]:
        if calendar.isleap(date.year):
            if timestep == "pentadal":
                if date.day in [5, 10, 15, 20, 25, 29]:
                    timestep_end = True
                else:
                    timestep_end = False
            elif timestep == "dekadal":
                if date.day in [10, 20, 29]:
                    timestep_end = True
                else:
                    timestep_end = False
            elif timestep == "monthly" or timestep == "seasonal":
                if date.day in [29]:
                    timestep_end = True
                else:
                    timestep_end = False
        else:
            if timestep == "pentadal":
                if date.day in [5, 10, 15, 20, 25, 28]:
                    timestep_end = True
                else:
                    timestep_end = False
            elif timestep == "dekadal":
                if date.day in [10, 20, 28]:
                    timestep_end = True
                else:
                    timestep_end = False
            elif timestep == "monthly" or timestep == "seasonal":
                if date.day in [28]:
                    timestep_end = True
                else:
                    timestep_end = False
    
    return timestep_end


def determine_daterange(startdate, enddate):
    """Determine files to download given supplied start/end dates.
    
    Parameters
    ----------
    startdate : str
        Start date of rainfall estimate (format: YYYY-MM-DD).
    enddate : str
        End date of the rainfall estimate (format: YYYY-MM-DD).
    
    Returns
    -------
    list
        Date range from start to end.
    
    """
    # Determine dates to download
    startdate = dt.strptime(startdate, "%Y-%m-%d")
    enddate = dt.strptime(enddate, "%Y-%m-%d")
    daterange = [startdate + td(n) for n in range(int((enddate - startdate).days + 1))]
    
    return(daterange)
    
    
def rfe_fname_constructor(timestep, url, version, date, degrade_tag, resolution):
    """Deduce RFE filename.
    
    Parameters
    ----------
    timestep : str
        Timestep - choices are 'daily','pentadal', 'dekadal' and 'monthly'.
    url : str
        TAMSAT data URL.
    version : str
        Version of TAMSAT rainfall estimates.
    date : str
        Date of file (for daily use YYYY-MM-DD, for pentadal use YYYY-MM-P
                      for dekadal use YYYY-MM-D and for monthly use YYYY-MM).
    degrade_tag : int
        Value is 0 if resolution is 0.0375 deg, otherwise value is 1.
    resolution : str
        Resolution of the TAMSAT estimates.
    
    Returns
    -------
    str
        TAMSAT filename conforming to input arguments.
    
    """
    date = date.replace('-', '')
    yyyy = date[0:4]
    mm = date[4:6]
    if timestep == 'daily':
        fname_str = f'rfe{yyyy}_{mm}_{date[6:8]}.v{version}.nc'
    elif timestep == 'pentadal':
        fname_str = f'rfe{yyyy}_{mm}-pt{date[6]}.v{version}.nc'
    elif timestep == 'pentadal-anomalies':
        fname_str = f'rfe{yyyy}_{mm}-pt{date[6]}_anom.v{version}.nc'
    elif timestep == 'dekadal':
        fname_str = f'rfe{yyyy}_{mm}-dk{date[6]}.v{version}.nc'
    elif timestep == 'dekadal-anomalies':
        fname_str = f'rfe{yyyy}_{mm}-dk{date[6]}_anom.v{version}.nc'
    elif timestep == 'monthly':
        fname_str = f'rfe{yyyy}_{mm}.v{version}.nc'
    elif timestep == 'monthly-anomalies':
        fname_str = f'rfe{yyyy}_{mm}_anom.v{version}.nc'
    elif timestep == 'seasonal':
        fname_str = f'rfe{yyyy}_{mm}_seas.v{version}.nc'
    elif timestep == 'seasonal-anomalies':
        fname_str = f'rfe{yyyy}_{mm}_seas_anom.v{version}.nc'
    
    fullpath = url + '/' + 'v' + version + '/' + timestep + '/' + str(yyyy) + '/' + str(mm) + '/' + fname_str
    
    if degrade_tag == 1:
        fname_str = fname_str.split(f'.v{version}')[0] + '_' + str(resolution) + f'.v{version}' + fname_str.split(f'.v{version}')[1]
        fullpath = url + '/' + 'v' + version + '/' + timestep + '/' + str(resolution) + '/' + str(yyyy) + '/' + str(mm) + '/' + fname_str
    
    return fullpath


def sm_fname_constructor(timestep, url, version, date):
    """Deduce SM filename.
    
    Parameters
    ----------
    timestep : str
        Timestep - choices are 'daily','pentadal', 'dekadal' and 'monthly'.
    url : str
        TAMSAT data URL.
    version : str
        Version of TAMSAT soil moisture estimates.
    date : str
        Date of file (for daily use YYYY-MM-DD, for pentadal use YYYY-MM-P
                      for dekadal use YYYY-MM-D and for monthly use YYYY-MM).
    
    Returns
    -------
    str
        TAMSAT filename conforming to input arguments.
    
    """
    date = date.replace('-', '')
    yyyy = date[0:4]
    mm = date[4:6]
    if timestep == 'daily':
        fname_str = f'sm{yyyy}_{mm}_{date[6:8]}.v{version}.nc'
    elif timestep == 'pentadal':
        fname_str = f'sm{yyyy}_{mm}-pt{date[6]}.v{version}.nc'
    elif timestep == 'pentadal-anomalies':
        fname_str = f'sm{yyyy}_{mm}-pt{date[6]}_anom.v{version}.nc'
    elif timestep == 'dekadal':
        fname_str = f'sm{yyyy}_{mm}-dk{date[6]}.v{version}.nc'
    elif timestep == 'dekadal-anomalies':
        fname_str = f'sm{yyyy}_{mm}-dk{date[6]}_anom.v{version}.nc'
    elif timestep == 'monthly':
        fname_str = f'sm{yyyy}_{mm}.v{version}.nc'
    elif timestep == 'monthly-anomalies':
        fname_str = f'sm{yyyy}_{mm}_anom.v{version}.nc'
    elif timestep == 'seasonal':
        fname_str = f'sm{yyyy}_{mm}_seas.v{version}.nc'
    elif timestep == 'seasonal-anomalies':
        fname_str = f'sm{yyyy}_{mm}_seas_anom.v{version}.nc'
    
    fullpath = url + '/' + 'v' + version + '/' + timestep + '/' + str(yyyy) + '/' + str(mm) + '/' + fname_str
    
    return fullpath


def get_filenames(remoteurl, daterange, timestep, version, resolution=None, product='rfe'):
    """Create list of TAMSAT filenames to download.
    
    Parameters
    ----------
    product: str
        TAMSAT data product – choices are 'rfe' or 'sm'.
    remoteurl : str
        TAMSAT data URL.
    daterange : list
        List of dates to consider.
    timestep : str
        Timestep - choices are 'pentadal', 'dekadal' and 'monthly'.
    resolution : str
        Spatial resolution - choices are '0.0375', '0.25', '0.50' and '1.00'.
    version : float
        Version of TAMSAT rainfall estimates.
    
    Returns
    -------
    list
        List of filenames to process.
    
    """

    if str(resolution) == '0.0375':
        degrade_tag = 0
    elif str(resolution) in ['0.25', '0.50', '1.00']:
        degrade_tag = 1
    
    if product == 'sm':
        dataurl = remoteurl + 'soil_moisture/data'
    elif degrade_tag == 0:
        dataurl = remoteurl + 'rfe/data'
    else:
        dataurl = remoteurl + 'rfe/data_degraded'
    
    files_to_download = []
    if timestep == 'daily':
        daily_dates = date_df('daily', dt.strftime(daterange[0], '%Y-%m-%d'), dt.strftime(daterange[-1], '%Y-%m-%d'), endmonth=False)
        daily_dates = daily_dates.Year.astype('str') + daily_dates.Month.astype('str') + daily_dates.Day.astype(str)
        if product == 'rfe':
            files_to_download = [rfe_fname_constructor(timestep, dataurl, str(version), x, degrade_tag, str(resolution)) for x in list(daily_dates)]
        elif product == 'sm':
            files_to_download = [sm_fname_constructor(timestep, dataurl, str(version), x) for x in list(daily_dates)]
    elif timestep == 'pentadal':
        pentad_dates = date_df('pentadal', dt.strftime(daterange[0], '%Y-%m-%d'), dt.strftime(daterange[-1], '%Y-%m-%d'), endmonth=False)
        pentad_dates = pentad_dates.Year.astype('str') + pentad_dates.Month.astype('str') + pentad_dates.Pentad.astype(str)
        if product == 'rfe':
            files_to_download = [rfe_fname_constructor(timestep, dataurl, str(version), x, degrade_tag, str(resolution)) for x in list(pentad_dates)]
        elif product == 'sm':
            files_to_download = [sm_fname_constructor(timestep, dataurl, str(version), x) for x in list(pentad_dates)]
    elif timestep == 'dekadal':
        dekad_dates = date_df('dekadal', dt.strftime(daterange[0], '%Y-%m-%d'), dt.strftime(daterange[-1], '%Y-%m-%d'), endmonth=False)
        dekad_dates = dekad_dates.Year.astype('str') + dekad_dates.Month.astype('str') + dekad_dates.Dekad.astype(str)
        if product == 'rfe':
            files_to_download = [rfe_fname_constructor(timestep, dataurl, str(version), x, degrade_tag, str(resolution)) for x in list(dekad_dates)]
        elif product == 'sm':
            files_to_download = [sm_fname_constructor(timestep, dataurl, str(version), x) for x in list(dekad_dates)]
    elif timestep == 'monthly':
        month_dates = date_df('monthly', dt.strftime(daterange[0], '%Y-%m-%d'), dt.strftime(daterange[-1], '%Y-%m-%d'), endmonth=False)
        month_dates = month_dates.Year.astype('str') + month_dates.Month.astype('str')
        if product == 'rfe':
            files_to_download = [rfe_fname_constructor(timestep, dataurl, str(version), x, degrade_tag, str(resolution)) for x in list(month_dates)]
        elif product == 'sm':
            files_to_download = [sm_fname_constructor(timestep, dataurl, str(version), x) for x in list(month_dates)]

    return(files_to_download)

def check_input_values(timestep, resolution, product='rfe'):
    """Check if timestep and resolution values are valid.
    
    Parameters
    ----------
    product: type
        TAMSAT product – choices are 'rfe' and 'sm'
    timestep : type
        Timestep - choices are 'pentadal', 'dekadal' and 'monthly'.
    resolution : type
        Spatial resolution - choices are '0.0375', '0.25', '0.50' and '1.00'.
    
    Returns
    -------
    bool
        True or False.
    
    """
    valid_products = ['rfe', 'sm']
    if product not in valid_products:
        product_check = False
        print("'product' not recognised - excepted values are 'rfe' and 'sm'")
    else:
        product_check = True

    valid_timesteps = ['daily', 'pentadal', 'dekadal', 'monthly']
    if timestep not in valid_timesteps:
        timestep_check = False
        print("'timestep' not recognised - excepted values are 'daily', 'pentadal', 'dekadal' or 'monthly'")
    else:
        timestep_check = True
    
    valid_resolutions = ['0.0375', '0.25', '0.50', '1.00']
    if str(resolution) not in valid_resolutions:
        resolution_check = False
        print("'resolution' not recognised - excepted values are '0.0375', '0.25', '0.50' and '1.00'")
    else:
        resolution_check = True
    
    if all(x for x in [product_check, timestep_check, resolution_check]):
        return True
    else:
        return False


def download_files(files_to_download, localdata_dir):
    """Download TAMSAT files.
    
    Parameters
    ----------
    files_to_download : list
        List of filenames to process
    localdata_dir : str
        Local path to store downloaded file.
    
    Returns
    -------
    Attempt to download file to local directory.
    
    """
    original_dir = os.getcwd()
    download_list = []
    for url_file in files_to_download:
        # Create directory for given file
        dirtmp = url_file.split('public')[1]
        localpath = localdata_dir + dirtmp
        
        # Check if local directory exists, if not, create
        if not os.path.exists(os.path.dirname(localpath)):
            os.makedirs(os.path.dirname(localpath))
        
        # Only download if file does not exist locally
        if not os.path.exists(localpath):
            download_list.append(url_file)
    
    print('%s file(s) to download' % len(download_list))
    if len(download_list) > 0:
        for url_file in download_list:
            dirtmp = url_file.split('public')[1]
            localpath = localdata_dir + dirtmp
            os.chdir(os.path.dirname(localpath))
            try:
                filename = wget.download(url_file)
                print(' Downloaded file: %s' % filename)
            except:
                print(' Unable to download file: %s' % url_file)
    os.chdir(original_dir)


def check_lonlat(lon, lat):
    """Check that supplied lon and lat are valid.
    
    Parameters
    ----------
    lon : float
        Longitude.
    lat : float
        Latitude.
    
    Returns
    -------
    bool
        True or False.
    
    """
    if (lon < -19.0125) or (lon > 51.975):
        print('Supplied longitude value is outside of TAMSAT domain, must be between -19.0125 and 51.975')
        lon_check = False
    else:
        lon_check = True
    
    if (lat < -35.9625) or (lat > 38.025):
        print('Supplied latitude value is outside of TAMSAT domain, must be between -35.9625 and 38.025')
        lat_check = False
    else:
        lat_check = True
    
    if all(x for x in [lon_check, lat_check]):
        return True
    else:
        return False
        
    
def check_dates(startdate, enddate):
    """Check that supplied dates are valid.
    
    Parameters
    ----------
    startdate : str
        Start date of rainfall estimate (format: YYYY-MM-DD).
    enddate : str
        End date of the rainfall estimate (format: YYYY-MM-DD).
    
    Returns
    -------
    bool
        True or False.
    
    """
    try:
        dt.strptime(startdate, "%Y-%m-%d")
        startdate_check = True
    except ValueError:
        print("'startdate' is incorrect. It should be YYYY-MM-DD")
        startdate_check = False
    
    try:
        dt.strptime(enddate, "%Y-%m-%d")
        enddate_check = True
    except ValueError:
        print("'enddate' is incorrect. It should be YYYY-MM-DD")
        enddate_check = False
    
    if all(x for x in [startdate_check, enddate_check]):
        return True
    else:
        return False


def check_domain(N, S, W, E):
    """Check that supplied coordinates are valid.
    
    Parameters
    ----------
    N, S, W, E : float
        North, South, West, East coordinates.
    
    """
    if (W < -19.0125) or (W > 51.975):
        print('Supplied "W" value is outside of TAMSAT domain, must be between -19.0125 and 51.975')
        W_check = False
    else:
        W_check = True
    
    if (E < -19.0125) or (E > 51.975):
        print('Supplied "E" value is outside of TAMSAT domain, must be between -19.0125 and 51.975')
        E_check = False
    else:
        E_check = True
    
    if (N < -35.9625) or (N > 38.025):
        print('Supplied "N" value is outside of TAMSAT domain, must be between -35.9625 and 38.025')
        N_check = False
    else:
        N_check = True
    
    if (S < -35.9625) or (S > 38.025):
        print('Supplied "S" value is outside of TAMSAT domain, must be between -35.9625 and 38.025')
        S_check = False
    else:
        S_check = True
    
    if all(x for x in [W_check, E_check, N_check, S_check]):
        return True
    else:
        return False


def download(request):
    """Handle download tasks.
    
    Parameters
    ----------
    request : dictionary
        Dictionary contain download arguments.
    
    Returns
    -------
    Attempt to download file to local directory.
    
    """
    product = request['product']
    timestep = request['timestep']
    resolution = request['resolution']
    startdate = request['start_date']
    enddate = request['end_date']
    version = request['version']
    localdata_dir = request['localdata_dir']
    
    # Check if supplied variables are valid
    if check_input_values(timestep, resolution, product):
        
        # Determine date range given supplied start/end dates.
        daterange = determine_daterange(startdate, enddate)
        
        # Get list of files to download
        files_to_download = get_filenames(remoteurl, daterange, timestep, version, resolution, product)
        
        # Download files
        download_files(files_to_download, localdata_dir)


def extract(request):
    """Extract TAMSAT rainfall for a given point, area or domain.
    
    Parameters
    ----------
    request : dictionary
        Dictionary contain download arguments.
    
    Returns
    -------
    Attempt to extract TAMSAT data for given arguments.
    
    """
    extract_type = request['extract_type']
    if extract_type in ['point', 'area_average', 'domain']:
        if extract_type == 'point':
            if 'longitude' not in request:
                print('Warning! "longitude" not supplied.')
                return
            
            if 'latitude' not in request:
                print('Warning! "latitude" not supplied.')
                return
            
            lon = float(request['longitude'])
            lat = float(request['latitude'])
            
            # Check if lon/lat values are valid
            if check_lonlat(lon, lat):
                pass
            else:
                return
            
            print('Extracting point TAMSAT estimates for longitude: %s and latitude: %s' % (lon, lat))
            
        elif (extract_type == 'area_average') or (extract_type == 'domain'):
            if 'N' not in request:
                print('Warning! "N" not supplied.')
                return
            
            if 'S' not in request:
                print('Warning! "S" not supplied.')
                return
            
            if 'W' not in request:
                print('Warning! "W" not supplied.')
                return
            
            if 'E' not in request:
                print('Warning! "E" not supplied.')
                return
            
            N = float(request['N'])
            S = float(request['S'])
            W = float(request['W'])
            E = float(request['E'])

            # Check if N/S/W/E values are valid
            if check_domain(N, S, W, E):
                pass
            else:
                return
            
            print('Extracting %s TAMSAT rainfall estimates for N: %s, S: %s, W: %s and E: %s' % (extract_type, str(N), str(S), str(W), str(E)))
        
        product = request['product']
        if 'product' not in request:
            product = 'rfe'

        timestep = request['timestep']
        if 'resolution' not in request:
            resolution = 0.25
        else:
            resolution = request['resolution']
        
        # Check if timestep and resolution values are valid
        if check_input_values(timestep, resolution, product):
            pass
        else:
            return
        
        startdate = request['start_date']
        enddate = request['end_date']
        
        # Check if start and end dates are valid
        if check_dates(startdate, enddate):
            pass
        else:
            return
        
        version = request['version']
        
        localdata_dir = request['localdata_dir']
        
        # List expected files
        daterange = determine_daterange(startdate, enddate)
        flist_expect = get_filenames(localdata_dir + '/tamsat/', daterange, timestep, version, resolution, product)
        
        # List files that exist
        flist_exist = [f for f in flist_expect if os.path.exists(f)]
        if len(flist_exist) > 0:
            if len(flist_exist) != len(flist_expect):
                print('Warning! Not all files within date range found: %s expected, %s found.' % (len(flist_expect), len(flist_exist)))
        
        # Extract
        if len(flist_exist) > 0:
            ds_list = []
            for file in flist_exist:
                ds = xr.open_dataset(file)
                if extract_type == 'point':
                    ds_list.append(ds.sel(lon=lon, lat=lat, method='nearest'))
                elif (extract_type == 'area_average') or (extract_type == 'domain'):
                    if str(resolution) == '0.0375':
                        ds_list.append(ds.sel(lon=slice(W, E), lat=slice(N, S)))
                    else:
                        ds_list.append(ds.sel(lon=slice(W, E), lat=slice(S, N)))
                
                ds.close()
            
            if len(ds_list) > 0:
                if extract_type == 'point':
                    ds = xr.concat(ds_list, dim='time')
                elif extract_type == 'area_average':
                    ds = xr.concat(ds_list, dim='time').mean(dim=['lon', 'lat'], skipna=True)
                elif extract_type == 'domain':
                    ds = xr.concat(ds_list, dim='time')
                
                if extract_type == 'point':
                    df = ds.to_dataframe().round(4)
                    fname = 'TAMSATv' + str(version) + '_' + timestep + '_' + str(resolution) + '_' + str(lon) + '_' + str(lat) + '_' + startdate + '_' + enddate + '.csv'
                    fname_full = localdata_dir + '/extracted_data/' + extract_type + '/' + fname
                    if not os.path.exists(os.path.dirname(fname_full)):
                        os.makedirs(os.path.dirname(fname_full))
                    
                    df.to_csv(fname_full, index=True, header=True)
                    
                elif extract_type == 'area_average':
                    df = ds.to_dataframe().round(4)
                    fname = 'TAMSATv' + str(version) + '_' + timestep + '_' + str(resolution) + '_' + str(N) + '_' + str(S) + '_' + str(W) + '_' + str(E) + '_' + startdate + '_' + enddate + '.csv'
                    fname_full = localdata_dir + '/extracted_data/' + extract_type + '/' + fname
                    if not os.path.exists(os.path.dirname(fname_full)):
                        os.makedirs(os.path.dirname(fname_full))
                    
                    df.to_csv(fname_full, index=True, header=True)
                    
                elif extract_type == 'domain':
                    fname = 'TAMSATv' + str(version) + '_' + timestep + '_' + str(resolution) + '_' + str(N) + '_' + str(S) + '_' + str(W) + '_' + str(E) + '_' + startdate + '_' + enddate + '.nc'
                    fname_full = localdata_dir + '/extracted_data/' + extract_type + '/' + fname
                    if not os.path.exists(os.path.dirname(fname_full)):
                        os.makedirs(os.path.dirname(fname_full))
                    
                    ds.to_netcdf(fname_full)
                
                if os.path.exists(fname_full):
                    print('Created file: %s' % fname_full)
                else:
                    print('Warning! Unable to create file: %s' % fname_full)
        else:
            print('No files found, please check input parameters or that TAMSAT data exists for given parameters.')
            print('By default, 0.25 degree resolution data are used for extraction unless "resolution" argument is supplied.')
    else:
        print('Warning! "extract_type" not recognised. Excepted values are: "point", "area_average" or "domain".')


# TAMSAT data URL
remoteurl = 'https://gws-access.jasmin.ac.uk/public/tamsat/'
