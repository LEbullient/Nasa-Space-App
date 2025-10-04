import math
from datetime import datetime
import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calc_speed(lat1, lon1, t1, lat2, lon2, t2):
    distance_km = haversine(lat1, lon1, lat2, lon2)

    time_diff_sec = (t2 - t1).total_seconds()
    if (time_diff_sec == 0):
        return distance_km, np.nan, np.nan

    speed_m_s = (distance_km * 1000) / time_diff_sec  # m/s
    speed_km_h = speed_m_s * 3.6                       # km/h

    return distance_km, speed_m_s, speed_km_h

MONTHS_2016 = []
for m in range(1, 13):
    ts = pd.Timestamp(f"2016-{m:02d}-15")
    sst_file = f"./train_data/sst/AQUA_MODIS.2016{m:02d}01_2016{m:02d}28.L3m.MO.NSST.sst.4km.nc"
    chl_file = f"./train_data/chl/AQUA_MODIS.2016{m:02d}01_2016{m:02d}28.L3m.MO.CHL.chlor_a.4km.nc"
    poc_file = f"./train_data/poc/AQUA_MODIS.2016{m:02d}01_2016{m:02d}28.L3m.MO.POC.poc.4km.nc"
    if m in [1, 3, 5, 7, 8, 10, 12]:
        end_day = 31
    elif m == 2:
        end_day = 29
    else:
        end_day = 30
    sst_file = f"./train_data/sst/AQUA_MODIS.2016{m:02d}01_2016{m:02d}{end_day}.L3m.MO.NSST.sst.4km.nc"
    chl_file = f"./train_data/chl/AQUA_MODIS.2016{m:02d}01_2016{m:02d}{end_day}.L3m.MO.CHL.chlor_a.4km.nc"
    poc_file = f"./train_data/poc/AQUA_MODIS.2016{m:02d}01_2016{m:02d}{end_day}.L3m.MO.POC.poc.4km.nc"
    MONTHS_2016.append({
        "month": m,
        "TS": ts,
        "SST_FILE": sst_file,
        "CHL_FILE": chl_file,
        "POC_FILE": poc_file
    })