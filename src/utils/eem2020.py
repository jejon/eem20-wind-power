import xarray as xr
import os
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch.nn as nn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path


def load_NWP(start="20000101", end="20001231"):
    """
    :param start: string start date (format: YYYYMMDD)
    :param end: string end date (format: YYYYMMDD)
    """
    files= list(pd.date_range(start=start, end=end, freq="D").strftime("%Y%m%d") + "T00Z.nc")
    dirname = str(Path.cwd().parents[1]) + "/data/eem20/raw"
    
    ds = xr.open_dataset(os.path.join(dirname,files.pop(0)))
    ds = ds.drop_vars("CloudCover")
    ds = ds.mean(dim="ensemble_member")
    #ds = ds.drop_vars("CloudCover").isel(y=slice(60,120), x=slice(11,71))
        
    for day in files:
        if os.path.isfile(os.path.join(dirname,day)):
            temp_ds = xr.open_dataset(os.path.join(dirname,day))
            temp_ds = temp_ds.drop_vars("CloudCover")
            temp_ds = temp_ds.mean(dim="ensemble_member")
            #temp_ds = temp_ds.drop_vars("CloudCover").isel(y=slice(60,120), x=slice(11,71))
            ds = ds.combine_first(temp_ds)
    return ds


def load_wind_power(start="20000101", end="20001231", adjust=True):
    start_date = start+"000000"
    end_date = end +"230000"
    dirname = str(Path.cwd().parents[1]) + "/data/eem20/processed"
    filename = "windpower_all.csv"
    df = pd.read_csv(os.path.join(dirname,filename),index_col=[0])
    df.index = df.index.astype("datetime64[ns]")
    df = df.loc[start_date:end_date]
    if adjust:
        if (np.where(df.index==pd.date_range("20000514", "20000514",freq="D")[0])[0]!=0):
            df = df.drop(pd.date_range(start="2000-05-14 00:00:00", end = "2000-05-14 23:00:00", freq="H"))
        if (np.where(pd.date_range("20000926", "20000926",freq="D")[0]==df.index)[0]!=0):
            df = df.drop(pd.date_range(start="2000-09-26 00:00:00", end = "2000-09-26 23:00:00", freq="H"))
        if (np.where(pd.date_range("20010730", "20010730",freq="D")[0]==df.index)[0]!=0):
            df = df.drop(pd.date_range(start="2001-07-30 00:00:00", end = "2001-07-30 23:00:00", freq="H"))
    ds = xr.Dataset(df)
    return ds


def load_wind_power_SE(start="20000101", end="20001231",SE="SE3"):
    power_data = load_wind_power(start, end)
    power_data = power_data.to_dataframe()
    if SE == "SE1":
        df_SE = pd.DataFrame(power_data["SE1"]).rename(columns={"SE1":"power"})
    elif SE == "SE2":
        df_SE = pd.DataFrame(power_data["SE2"]).rename(columns={"SE2":"power"})
    elif SE == "SE3":
        df_SE = pd.DataFrame(power_data["SE3"]).rename(columns={"SE3":"power"})
    elif SE == "SE4":
        df_SE = pd.DataFrame(power_data["SE4"]).rename(columns={"SE4":"power"})
    return df_SE


def load_X_dense(start="20000101", end="20001231", SE="SE3"):
    start_date = start
    end_date = end +   "230000"
    dates = pd.date_range(start=start_date, end=end_date, freq="H")
    dates = pd.DataFrame(dates,columns=["datetime"])
    dates.index = dates.datetime.astype("datetime64[ns]")
    if (np.where(dates==pd.date_range("20000514", "20000514",freq="D")[0])[0]!=0):
        dates = dates.drop(pd.date_range(start="2000-05-14 00:00:00", end = "2000-05-14 23:00:00", freq="H"))
    if (np.where(pd.date_range("20000926", "20000926",freq="D")[0]==dates)[0]!=0):
        dates = dates.drop(pd.date_range(start="2000-09-26 00:00:00", end = "2000-09-26 23:00:00", freq="H"))
    if (np.where(pd.date_range("20010730", "20010730",freq="D")[0]==dates)[0]!=0):
        dates = dates.drop(pd.date_range(start="2001-07-30 00:00:00", end = "2001-07-30 23:00:00", freq="H"))
    
    dates["day"] = dates["datetime"].dt.day
    dates["sin_day"] = np.sin(2*np.pi*dates.day/365)
    dates["cos_day"] = np.cos(2*np.pi*dates.day/365)
    dates["hour"] = dates["datetime"].dt.hour
    dates["sin_hour"] = np.sin(2*np.pi*dates.hour/24)
    dates["cos_hour"] = np.cos(2*np.pi*dates.hour/24)

    dates = dates.drop(columns=["datetime","day","hour"])
    
    return np.array(dates)


def load_max_power(start="20000101", end="20001231", SE="SE3", freq="H", adjust=True, SE1_adjusted=False):
    dates = pd.date_range(start=start, end=end, freq="D")
    if adjust:
        if (np.where(dates==pd.date_range("20000514", "20000514",freq="D")[0])[0]!=0):
            dates = dates.drop("20000514")
        if (np.where(pd.date_range("20000926", "20000926",freq="D")[0]==dates)[0]!=0):
            dates = dates.drop("20000926")
        if (np.where(pd.date_range("20010730", "20010730",freq="D")[0]==dates)[0]!=0):
            dates = dates.drop("20010730")
    df_wt = load_wind_turbines(SE1_adjusted)
    df_wt["Installation date"] = pd.to_datetime(df_wt["Installation date"])
    df_wt = df_wt[df_wt["Price region"]==SE]
    df_wt.sort_values("Installation date")
    df_wt["Cum. power"] = df_wt["Max power [MW]"].cumsum()
    
    cum_power = np.zeros((len(dates),1))
    for i, date in enumerate(dates):
        cum_power[i] = df_wt["Cum. power"][df_wt["Installation date"] <= date].max()
    if freq=="H":
        cum_power = np.repeat(cum_power,24,axis=0)
    return cum_power


def load_wind_power_TS(start="20000101", end="20001231"):
    power_data = load_wind_power(start=start,end=end).to_dataframe()
    dates = pd.date_range(start,end,freq="D")
    if (np.where(dates==pd.date_range("20000514", "20000514",freq="D")[0])[0]!=0):
        dates = dates.drop("20000514")
    if (np.where(pd.date_range("20000926", "20000926",freq="D")[0]==dates)[0]!=0):
        dates = dates.drop("20000926")
    if (np.where(pd.date_range("20010730", "20010730",freq="D")[0]==dates)[0]!=0):
        dates = dates.drop("20010730")
    TS = np.zeros([len(dates),4,24])
    for n in range(0,4):
        SEi = "SE" + str(n+1)
        power_data_SE = pd.DataFrame(power_data[SEi])
        for i in range(1,len(dates)):
            date = dates[i-1].date()
            ar = np.array(power_data_SE[str(date)]).flatten()
            TS[i,n] = ar
    TS = np.repeat(TS,24,axis=0)                  
    return TS


def load_wind_turbines(SE1_adjusted=True):
    if SE1_adjusted:
        dirname = str(Path.cwd().parents[1]) + "/data/eem20/processed"
        df = pd.read_excel(os.path.join(dirname,"turbines_adjusted_SE1.xlsx"),index_col=[0])
    else:
        dirname = str(Path.cwd().parents[1]) + "/data/eem20/processed"
        df = pd.read_csv(os.path.join(dirname,"windturbines_adjusted.csv"),index_col=[0])
    return df


def load_turbine_map(start="20000101",end="20001231", SE="SE3", norm=True, SE1_adjusted = True, close_one=False):
    ds_NWP = load_NWP(start="20000101", end="20000101")
    df_wt = load_wind_turbines(SE1_adjusted)
    df_wt["Installation date"] = pd.to_datetime(df_wt["Installation date"])
    df_wt = df_wt[df_wt["Price region"]==SE]
    grid_lat = np.array(ds_NWP.coords["latitude"].values)
    grid_lon = np.array(ds_NWP.coords["longitude"].values)
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    dates = pd.date_range(start=start, end=end, freq="D")
    grid_power = np.zeros([len(dates),169, 71])
    for _, turbine in df_wt.iterrows():
        lat = turbine["Latitude"]
        lon = turbine["Longitude"]
        result = abs(np.arccos(np.sin(grid_lat/(180/math.pi))*np.sin(lat/(180/math.pi)) + np.cos(grid_lat/(180/math.pi))*np.cos(lat/(180/math.pi))*np.cos((grid_lon/(180/math.pi))-(lon/(180/math.pi)))))
        if close_one:
            x, y = np.where(result == np.min(result))
            idx_dates = np.where(turbine["Installation date"]<=dates)[0]
            if idx_dates.size > 0:
                idx_date = idx_dates[0]
                for i in range(idx_date,len(dates)):
                    grid_power[i,x,y] += turbine["Max power [MW]"]
        else:
            sorted_result = np.partition(result.flatten(), 3)
            x1, y1 = np.where(result == sorted_result[0])
            x2, y2 = np.where(result == sorted_result[1])
            x3, y3 = np.where(result == sorted_result[2])
            x4, y4 = np.where(result == sorted_result[3])
            idx_dates = np.where(turbine["Installation date"]<=dates)[0]
            if idx_dates.size > 0:
                w1 = 1 - sorted_result[0] / (np.sum(sorted_result[:4]))
                w2 = 1 - sorted_result[1] / (np.sum(sorted_result[:4]))
                w3 = 1 - sorted_result[2] / (np.sum(sorted_result[:4]))
                w4 = 1 - sorted_result[3] / (np.sum(sorted_result[:4]))
                idx_date = idx_dates[0]
                for i in range(idx_date,len(dates)):
                    grid_power[i,x1,y1] += w1*turbine["Max power [MW]"]
                    grid_power[i,x2,y2] += w2*turbine["Max power [MW]"]
                    grid_power[i,x3,y3] += w3*turbine["Max power [MW]"]
                    grid_power[i,x4,y4] += w4*turbine["Max power [MW]"]
        
    if np.where(pd.date_range("20000514", "20000514",freq="D")[0]==dates)[0].size > 0:
        grid_power = np.delete(grid_power,np.where(pd.date_range("20000514", "20000514",freq="D")[0]==dates)[0][0],0)
    if np.where(pd.date_range("20000926", "20000926",freq="D")[0]==dates)[0].size > 0:
        grid_power = np.delete(grid_power,np.where(pd.date_range("20000514", "20000514",freq="D")[0]==dates)[0][0],0)
    if np.where(pd.date_range("20010730", "20010730",freq="D")[0]==dates)[0].size > 0:
        grid_power = np.delete(grid_power,np.where(pd.date_range("20010730", "20010730",freq="D")[0]==dates)[0][0],0)
    
    if norm:
        max_power = load_max_power(start=start, end=end, SE=SE, freq="D")
        for i in range(len(grid_power)):
            grid_power[i] /= max_power[i]
    return grid_power


def bin_power_values(power_series, nb_bins=256):
    bins = np.linspace(0,1,nb_bins)
    bin_idxs = np.digitize(power_series,bins)
    bin_idxs -= 1
    return bin_idxs


def special_split(dataset, shift=0, days_in_val=2, period=15):
    number_of_days = int(len(dataset) / 24)
    indices_train = [n for n in range(number_of_days) if ((n + shift*days_in_val) % period) < (period - days_in_val)]
    indices_train = [day*24 + hour for day in indices_train for hour in range(24)]
    indices_val = [n for n in range(number_of_days) if ((n + shift*days_in_val) % period) >= (period - days_in_val)]
    indices_val = [day*24 + hour for day in indices_val for hour in range(24)]
    return torch.utils.data.Subset(dataset, indices_train), torch.utils.data.Subset(dataset, indices_val)


def special_split_all_regions(dataset, shift=0, days_in_val=2, period=15):
    number_of_days = int(len(dataset) / (4*24))
    indices_train = [n for n in range(number_of_days) if ((n + shift*days_in_val) % period) < (period - days_in_val)]
    indices_train = [SE*number_of_days*24 + day*24 + hour for day in indices_train for hour in range(24) for SE in range(4)]
    indices_val = [n for n in range(number_of_days) if ((n + shift*days_in_val) % period) >= (period - days_in_val)]
    indices_val = [SE*number_of_days*24 + day*24 + hour for day in indices_val for hour in range(24) for SE in range(4)]
    return torch.utils.data.Subset(dataset, indices_train), torch.utils.data.Subset(dataset, indices_val)


class PinBallLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert target.size(1) == 1
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[...,0] - preds[...,i]
            losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(-1))
        return torch.cat(losses,1).mean()


def MAPE(output, target):
    return torch.mean(torch.abs((target-output)/target))


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    COPIED FROM: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(self, patience=25, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # Sava model
            torch.save(model.state_dict(), "best_weights.pt")
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class EEM20DatasetFinal(Dataset):
    def __init__(self, start="20000101", end="20001231", testset=False, NWP_mean = None, NWP_std = None):
        """
        :param start: string start date (format: YYYYMMDD)
        :param end: string end date (format: YYYYMMDD)
        """
        DIRPATH = str(Path.cwd().parents[1])
        self.SE_cat = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        self.power_data = load_wind_power(start,end).to_dataframe()
        self.time = self.power_data.index
        self.power_data = np.array(self.power_data)
        self.max_power = np.concatenate((load_max_power(start=start, end=end, SE="SE1", SE1_adjusted=True),
                                         load_max_power(start=start, end=end, SE="SE2"),
                                         load_max_power(start=start, end=end, SE="SE3"),
                                         load_max_power(start=start, end=end, SE="SE4")), axis=1)
        self.power_data_norm = self.power_data / self.max_power
        
        if not testset:
            self.NWP_data = xr.open_dataset(DIRPATH + "/data/eem20/processed/EEM2020_2000.nc")
        else:    
            self.NWP_data = load_NWP(start,end)
        self.NWP_data = self.NWP_data.assign(Wind=np.sqrt(self.NWP_data.Wind_U**2 + self.NWP_data.Wind_V**2))
        self.NWP_data = self.NWP_data.drop_vars(["Wind_U", "Wind_V","Temperature", "Pressure", "RelativeHumidity"])
        if NWP_mean is None:
            self.NWP_mean = self.NWP_data.mean()
            self.NWP_std = self.NWP_data.std()
        else:
            self.NWP_mean = NWP_mean
            self.NWP_std = NWP_std
        self.NWP_data = (self.NWP_data - self.NWP_mean) / self.NWP_std
        self.X_data_SE1 = self.NWP_data.isel(y=slice(92,156), x=slice(7,71))
        self.map_tur = load_turbine_map(start, end, "SE1", norm=True, SE1_adjusted=True, close_one=False)[:,92:156,7:]
        self.map_tur = xr.DataArray(np.repeat(self.map_tur, 24, axis=0), dims=["time","y","x"])
        self.NWP_ensemble_std_SE1 = xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_ensemble_std_SE1.nc").drop_vars(["Wind_U", "Wind_V","Temperature", "Pressure", "RelativeHumidity", "WindDensity"]).isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.NWP_mean_temp_SE1 =  xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_mean_temp_SE1.nc").isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.X_data_SE1 = self.X_data_SE1.assign(power=self.map_tur).to_array().transpose('time', 'variable', 'y', 'x')
        self.X_data_SE2 = self.NWP_data.isel(y=slice(58,122), x=slice(7,71))
        self.map_tur = load_turbine_map(start, end, "SE2", norm=True, close_one=False)[:,58:122,7:71]
        self.map_tur = xr.DataArray(np.repeat(self.map_tur, 24, axis=0), dims=["time","y","x"])
        self.NWP_ensemble_std_SE2 = xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_ensemble_std_SE2.nc").drop_vars(["Wind_U", "Wind_V","Temperature", "Pressure", "RelativeHumidity", "WindDensity"]).isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.NWP_mean_temp_SE2 =  xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_mean_temp_SE2.nc").isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.X_data_SE2 = self.X_data_SE2.assign(power=self.map_tur).to_array().transpose('time', 'variable', 'y', 'x')
        self.X_data_SE3 = self.NWP_data.isel(y=slice(14,78), x=slice(1,65))
        self.map_tur = load_turbine_map(start, end, "SE3", norm=True, close_one=False)[:,14:78,1:65]
        self.map_tur = xr.DataArray(np.repeat(self.map_tur, 24, axis=0), dims=["time","y","x"])
        self.NWP_ensemble_std_SE3 = xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_ensemble_std_SE3.nc").drop_vars(["Wind_U", "Wind_V","Temperature", "Pressure", "RelativeHumidity", "WindDensity"]).isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.NWP_mean_temp_SE3 =  xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_mean_temp_SE3.nc").isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.X_data_SE3 = self.X_data_SE3.assign(power=self.map_tur).to_array().transpose('time', 'variable', 'y', 'x')
        self.X_data_SE4 = self.NWP_data.isel(y=slice(0,64), x=slice(0,64))
        self.map_tur = load_turbine_map(start, end, "SE4", norm=True, close_one=False)[:,0:64,0:64]
        self.map_tur = xr.DataArray(np.repeat(self.map_tur, 24, axis=0), dims=["time","y","x"])
        self.X_data_SE4 = self.X_data_SE4.assign(power=self.map_tur).to_array().transpose('time', 'variable', 'y', 'x')
        self.NWP_ensemble_std_SE4 = xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_ensemble_std_SE4.nc").drop_vars(["Wind_U", "Wind_V","Temperature", "Pressure", "RelativeHumidity", "WindDensity"]).isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.NWP_mean_temp_SE4 =  xr.open_dataset(DIRPATH + "/data/eem20/processed/NWP_mean_temp_SE4.nc").isel(time=slice(0,len(self.power_data))).to_array().transpose('time', 'variable')
        self.start = start
        self.end = end
    
    def __len__(self):
        return len(self.power_data)*4
    
    def __getitem__(self, index):
        time_index = index % len(self.power_data)
        SE_index = index // len(self.power_data)
        
        y_norm = torch.tensor(self.power_data_norm[time_index, SE_index])
        max_power = torch.tensor(self.max_power[time_index, SE_index])
        SE_cat = torch.tensor(self.SE_cat[SE_index])
        dayofweek = self.time[time_index].day_of_week
        dayofweek_vec = torch.zeros(7)
        dayofweek_vec[dayofweek] = 1
        hour = self.time[time_index].hour
        hour_vec = torch.zeros(24)
        hour_vec[hour] = 1
        if SE_index == 0:
            X = torch.tensor(self.X_data_SE1[time_index].values)
            NWP_ensemble_std = torch.tensor(self.NWP_ensemble_std_SE1[time_index].values)
            NWP_mean_temp = torch.tensor(self.NWP_mean_temp_SE1[time_index].values)
        elif SE_index == 1:
            X = torch.tensor(self.X_data_SE2[time_index].values)
            NWP_ensemble_std = torch.tensor(self.NWP_ensemble_std_SE2[time_index].values)
            NWP_mean_temp = torch.tensor(self.NWP_mean_temp_SE2[time_index].values)
        elif SE_index == 2:
            X = torch.tensor(self.X_data_SE3[time_index].values)
            NWP_ensemble_std = torch.tensor(self.NWP_ensemble_std_SE3[time_index].values)
            NWP_mean_temp = torch.tensor(self.NWP_mean_temp_SE3[time_index].values)
        else:
            X = torch.tensor(self.X_data_SE4[time_index].values)
            NWP_ensemble_std = torch.tensor(self.NWP_ensemble_std_SE4[time_index].values)
            NWP_mean_temp = torch.tensor(self.NWP_mean_temp_SE4[time_index].values)
        
        X, SE_cat, hour_vec, dayofweek_vec, max_power, y_norm, NWP_ensemble_std, NWP_mean_temp = X.float(), SE_cat.float(), hour_vec.float(), dayofweek_vec.float(), max_power.float(), y_norm.float(), NWP_ensemble_std.float(), NWP_mean_temp.float()
        return X, SE_cat, hour_vec, dayofweek_vec, max_power, y_norm, NWP_ensemble_std, NWP_mean_temp, self.time[time_index].year, self.time[time_index].month, self.time[time_index].day, hour, dayofweek, SE_index+1