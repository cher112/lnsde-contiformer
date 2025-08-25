# 读取光变曲线数据
# 读取光变曲线数据，返回光变曲线数据的字典
import os

import joblib
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from func_timeout import func_set_timeout
from matplotlib import pyplot as plt
from tqdm import tqdm
from lightkurve import LightCurve as lk
from supersmoother import SuperSmoother
import warnings

warnings.filterwarnings('error')


class LightCurve:
    def __init__(self, times, measurements, errors, survey=None, length=0, period=None,
                 label=None, p_class=None, is_downsampling=True):
        self.times = times
        self.measurements = measurements
        self.errors = errors
        self.survey = survey
        self.period = period
        self.label = label
        self.p_class = p_class
        self.is_downsampling = is_downsampling
        self.length = length
        self.fit_lc = None

    def __len__(self):
        return self.length

    def lc_padding(self, length=256, mode='zero'):
        if self.length < length:
            padding_num = length - self.length
            self.times = np.pad(self.times, (0, padding_num), 'constant', constant_values=(0, -1e9))
            if mode == 'zero':
                self.measurements = np.pad(self.measurements, (0, padding_num), 'constant', constant_values=(0, 0))
            elif mode == 'mean':
                self.measurements = np.pad(self.measurements, (0, padding_num), 'constant',
                                           constant_values=(0, np.mean(self.measurements)))
            else:
                assert False, 'Invalid padding mode'
            self.errors = np.pad(self.errors, (0, padding_num), 'constant', constant_values=(0, 0))

    @func_set_timeout(5)
    def fit_lomb_scargle(self):
        try:
            frequency = np.linspace(0.001, 50, 500000)

            power = LombScargle(self.times, self.measurements, self.errors).power(frequency)
            self.period = 1 / frequency[np.argmax(power)]
            return 0
        except:
            return 1

    @func_set_timeout(5)
    def fit_supersmoother(self):
        try:
            ss = SuperSmoother()
            t_linspace = np.linspace(self.times.min(), self.times.max(), 256)

            ss.fit(self.times, self.measurements, self.errors)
            self.fit_lc = ss.predict(t_linspace)
            return 0
        except:
            return 1

    def fold(self, period, period_scale=2):
        lc = lk(time=self.times, flux=self.measurements, flux_err=self.errors)
        try:
            flc = lc.fold(period=period * period_scale)
        except:
            flc = lc

        self.times = flc.time.value
        self.measurements = flc.flux.value
        self.errors = flc.flux_err.value

    def downsampling(self, sample_num=256):
        idx = np.arange(self.length)
        new_idx = np.random.choice(idx, size=sample_num)
        self.times = self.times[new_idx]
        self.measurements = self.measurements[new_idx]
        self.errors = self.errors[new_idx]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.errorbar(self.times, self.measurements, yerr=self.errors, fmt='o')
        plt.title(self.label)
        plt.show()

    def delete_outlier(self, min_quantile=1, max_quantile=99):
        idx = np.intersect1d(np.argwhere(
            np.percentile(self.measurements, min_quantile) < self.measurements), np.argwhere(
            np.percentile(self.measurements, max_quantile) > self.measurements))
        if len(idx) == 0:
            return
        else:
            self.times = self.times[idx]
            self.measurements = self.measurements[idx]
            self.errors = self.errors[idx]
            self.length = len(self.times)


class LightCurveProcess:
    def __init__(self, label_map_path, survey):
        self.label_map_path = label_map_path
        self.survey = survey

    def load_data(self):
        root = os.path.abspath(os.path.join(os.getcwd()))
        print(root)
        data_root = os.path.join(root, 'dataset')
        label_map = pd.read_csv(self.label_map_path)
        if 'Unnamed: 0' in label_map.columns:
            label_map = label_map.drop(['Unnamed: 0'], axis=1)

        light_curve_list = []
        for i in tqdm(range(len(label_map))):
            if self.survey == 'LINEAR':
                data_path = 'LINEAR/' + label_map.iloc[i]['Path']
            elif self.survey == 'MACHO':
                data_path = 'MACHO/' + label_map.iloc[i]['Path']
            else:
                data_path = label_map.iloc[i]['Path']
            label = label_map.iloc[i]['Class']
            length = label_map.iloc[i]['N']
            data = pd.read_csv(os.path.join(data_root, data_path))

            if self.survey == 'ASAS' or self.survey == 'LINEAR' or self.survey == 'MACHO':
                times = data['mjd'].values
                measurements = data['mag'].values
                errors = data['errmag'].values
            elif self.survey == 'CSS':
                times = data['MJD'].values
                measurements = data['Mag'].values
                errors = data['Magerr'].values
            try:
                lc = LightCurve(times, measurements, errors, survey=self.survey, label=label, length=length)
                lc.delete_outlier()
                if lc.period is None:
                    lc.fit_lomb_scargle()
                period = lc.period
                lc.fold(period=period, period_scale=2)

                if lc.fit_supersmoother() == 1:
                    continue
                if len(lc) < 256:
                    lc.lc_padding(length=256, mode='mean')
                else:
                    lc.downsampling(sample_num=256)
            except:
                continue

            light_curve_list.append(lc)
        save_name = os.path.join(data_root, 'pkls', f'{self.survey}.pkl')
        joblib.dump(light_curve_list, filename=save_name, compress=3)


class ASASSNLightCurveProcess:
    def __init__(self, lc_path, catalog_path, survey):
        self.lc_path = lc_path
        self.catalog_path = catalog_path
        self.survey = survey

    def load_data(self):
        root = os.path.abspath(os.path.join(os.getcwd()))
        data_root = os.path.join(root, 'dataset')

        variables_catalogs = pd.read_csv(self.catalog_path, low_memory=False)
        # 选择class_prob大于0.95的类别
        variables_catalogs = variables_catalogs[variables_catalogs['class_probability'] > 0.95]

        # numpy统计每个类别的数量
        unique, counts = np.unique(variables_catalogs['variable_type'], return_counts=True)
        # 保存大于1000的类别
        unique = unique[counts > 2000]
        # 按照类别对数据进行筛选
        variables_catalogs = variables_catalogs[variables_catalogs['variable_type'].isin(unique)]

        light_curve_list = []
        for i in tqdm(range(len(variables_catalogs))):
            # for i in tqdm(range(2)):
            # 读取数据，并删除第一行
            name = 'ASASSN-V' + variables_catalogs.iloc[i]['asassn_name'][9:] + '.dat'
            data = pd.read_csv(os.path.join(self.lc_path, name),
                               skiprows=1, delim_whitespace=True, engine='c')
            period = variables_catalogs.iloc[i]['period']
            times = data['HJD'].values
            measurements = data['MAG'].values
            errors = data['MAG_ERR'].values
            length = len(times)
            try:
                lc = LightCurve(times, measurements, errors, survey=self.survey,
                                label=variables_catalogs.iloc[i]['variable_type'], period=period, length=length)
                lc.delete_outlier()
                if lc.period is None:
                    lc.fit_lomb_scargle()
                lc.fold(period=period, period_scale=2)
                lc.fit_supersmoother()
                if len(lc) < 256:
                    lc.lc_padding(length=256, mode='mean')
                else:
                    lc.downsampling(sample_num=256)
            except:
                continue
            light_curve_list.append(lc)
        # save_name = os.path.join(data_root, 'pkls', f'{self.survey}.pkl')
        save_dir = os.path.join(data_root, 'pkls')
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, f'{self.survey}.pkl')
        joblib.dump(light_curve_list, filename=save_name, compress=3)


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.getcwd()))
    # root = os.path.join(root, 'src')
    print(root)
    label_path = 'dataset/LINEAR/LINEAR_dataset.dat'
    light_curve_process = LightCurveProcess(os.path.join(root, label_path), survey='LINEAR')
    light_curve_process.load_data()

    # lc_path = 'dataset/ASASSN/'
    # catalog_path = 'dataset/asassn_catalog_full.csv'
    # light_curve_process = ASASSNLightCurveProcess(os.path.join(root, lc_path), os.path.join(root, catalog_path),
    #                                               survey='ASASSN')
    # light_curve_process.load_data()
