import ui
import os
import re
import sys
from PyQt5 import QtWidgets, QtCore
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


APATH = r'.\Afiles'
RESULTPATH = r'.\results'
RESTR = re.compile('^A\d{5}-\d{6}.txt$', re.IGNORECASE)
Ui_MainWindow = ui.Ui_Form


def get_stids():
    stids = [stid for stid in os.listdir(APATH) if os.path.isdir(os.path.join(APATH, stid))]
    return stids


def my_polar(wind_preps, title, result_dir, fig_name):
    wind_direc = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'C']
    wd_preps = []
    for direc in wind_direc:
        wd_preps.append(wind_preps[direc])

    # 只画16个方位的频率，C的频率以标题形式给出
    wd_len = len(wind_direc[:-1])

    prep_max = max(wd_preps) + 2
    preps = np.array(wd_preps[:-1])
    # 设置极坐标x的范围
    angles = np.linspace(0, 2 * np.pi, wd_len, endpoint=False)
    # 设置极坐标及频率数据起止点的闭合
    angles = np.concatenate((angles, [angles[0]]))
    preps = np.concatenate((preps, [preps[0]]))
    # 开始绘制风向玫瑰图
    fig = plt.figure(figsize=(15, 12))
    # 创建极坐标
    ax = fig.add_subplot(121, projection='polar')
    # 画线
    ax.plot(angles, preps, 'k-', linewidth=1)
    # 设置极坐标0°对应位置
    ax.set_theta_zero_location('N')
    # 设置极坐标正方向，1：逆时针，-1：顺时针
    ax.set_theta_direction(-1)
    ax.fill(angles, preps, facecolor='w', alpha=0.25)
    # 设置坐标轴显示
    ax.set_thetagrids(angles * 180 / np.pi, labels=wind_direc[:-1], fontsize=15, fontproperties="SimHei")
    # 设置极径网格线显示
    ax.set_rgrids(np.arange(2, prep_max, 2))
    # 设置极径的范围
    ax.set_rlim(0, prep_max)
    # 设置极径标签显示位置
    ax.set_rlabel_position('30')
    # 设置图题
    # ax.text(angles[0], prep_max + 4, title, horizontalalignment='center', verticalalignment='center', fontsize=15,
    #         fontproperties="SimHei", bbox=dict(facecolor='white'))
    ax.set_title(title, fontsize=12)
    ax.grid(True)
    plt.savefig(os.path.join(result_dir, fig_name), dpi=75)
    # plt.show()


class AparseWd:
    """
    参数可以为  ：字典，key为16方位及C，value为各方位对应的频率值
              或：A文件全路径
    """

    def __init__(self, afiles):
        self.afiles = afiles
        self.wind_direc = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'C']

    def parse_afile_wd(self, cate_str):
        """
        :return: 根据A文件返回当月各风向的出现频率
        """
        datas = {}
        columns = ['月'] + self.wind_direc
        for afile in self.afiles:
            wind_preps = {}
            year_month = os.path.split(afile)[-1][7: 13]
            with open(afile, 'r') as f:
                while True:
                    buf = f.readline()
                    if 'FN' in buf:
                        break
                while True:
                    buf = f.readline().strip()
                    for b in buf.split():
                        if '=' in b or '.' in b:
                            b = b[:-1]
                        try:
                            ws = int(b[3:]) / 10
                        except ValueError:
                            continue
                        else:
                            if ws <= 0.2:
                                wd = 'C'
                            else:
                                wd = int(b[:3])
                                wd_loc = int((wd + 11.25) // 22.5)
                                if wd_loc > 15:
                                    wd = 'N'
                                else:
                                    wd = self.wind_direc[wd_loc]
                            wind_preps.setdefault(wd, 0)
                            wind_preps[wd] += 1
                    if '=' in buf:
                        break
            wd_total_num = sum(list(wind_preps.values()))

            for w in self.wind_direc:
                wind_preps[w] = int(round(wind_preps[w] / wd_total_num, 2) * 100)
            wind_preps['月'] = int(year_month[4:])
            datas[year_month] = wind_preps
        pd_datas = pd.DataFrame(datas, index=columns)
        results = pd.DataFrame(pd_datas.T, columns=columns)
        if cate_str == 'ave':
            results = results.groupby('月').mean()
            results = pd.DataFrame(results)
        if cate_str == 'every':
            results = results.iloc[:, 1:]
            results.index.name = '年月'
        return results


class ParseDate:
    def __init__(self, begin_date, end_date):
        self.begin_year, self.begin_month = begin_date[:4], begin_date[4:]
        self.end_year, self.end_month = end_date[:4], end_date[4:]

    def parse_for_list(self):
        date_list = []
        dt = self.begin_year + self.begin_month
        while dt <= self.end_year + self.end_month:
            date_list.append(dt)
            buf_month = str(int(self.begin_month) + 1)
            if len(buf_month) < 2:
                buf_month = '0' + buf_month
            if int(buf_month) > 12:
                self.begin_month = '01'
                self.begin_year = str(int(self.begin_year) + 1)
            else:
                self.begin_month = buf_month
                self.begin_year = self.begin_year
            dt = self.begin_year + self.begin_month
        return date_list


class MyApp(QtWidgets.QDialog, Ui_MainWindow):
    # 这里的第一个变量是你该窗口的类型，第二个是该窗口对象。
    def __init__(self):
        # 创建主界面对象
        QtWidgets.QDialog.__init__(self)
        # 主界面对象初始化
        Ui_MainWindow.__init__(self)
        # 配置主界面对象
        self.setupUi(self)

        defalut_date = datetime.date.today()
        # 设置开始时间, 并设置最大输入时间
        self.StratDate.setDateTime(
            QtCore.QDateTime(QtCore.QDate(defalut_date.year, defalut_date.month, 1),
                             QtCore.QTime(0, 0, 0)))
        self.StratDate.setMaximumDateTime(
            QtCore.QDateTime(QtCore.QDate(defalut_date.year, defalut_date.month, 1),
                             QtCore.QTime(0, 0, 0)))
        # 设置结束时间， 并设置最大输入时间
        self.EndDate.setDateTime(
            QtCore.QDateTime(QtCore.QDate(defalut_date.year, defalut_date.month, 1),
                             QtCore.QTime(0, 0, 0)))
        self.EndDate.setMaximumDateTime(
            QtCore.QDateTime(QtCore.QDate(defalut_date.year, defalut_date.month, 1),
                             QtCore.QTime(0, 0, 0)))
        self.StSelect.addItems(get_stids())

        self.ResultBotton.clicked.connect(self.wdpolars)

    def wdpolars(self):
        stid = self.StSelect.currentText()
        stpath = os.path.join(APATH, stid)

        start_date = self.StratDate.date().toString('yyyyMM')
        end_date = self.EndDate.date().toString('yyyyMM')

        if self.EveryResult.isChecked():
            cate_str = 'every'

        else:
            cate_str = 'ave'
        title_str = ''.join(['{}-', '风向频率图 ', '静风频率:', '{}'])

        if end_date < start_date:
            msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning,
                                            "Alert", "结束时间不能早于开始时间,请重新输入!")
            msg_box.exec_()
        else:
            date_parsed = ParseDate(start_date, end_date)
            dates = date_parsed.parse_for_list()
            afiles = [os.path.join(stpath, afile) for afile in os.listdir(stpath)
                      if RESTR.match(afile) and afile[7: 13] in dates]
            if len(afiles) < len(dates):
                msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Alert", "缺少部分A文件，请检查！")
                msg_box.exec_()
            else:
                wind_preps = AparseWd(afiles).parse_afile_wd(cate_str)
                wind_preps.to_excel('{}风向频率统计结果.xlsx'.format(stid + '_' + start_date + '-' + end_date))
                # print(wind_preps)
                os.makedirs(RESULTPATH, exist_ok=True)
                for n, result_date in enumerate(list(wind_preps.index)):
                    preps = dict(wind_preps.iloc[n, :])
                    # print(preps)
                    if len(result_date) >= 2:
                        result_date = ''.join([result_date[:4], '年', result_date[4:], '月'])
                    else:
                        result_date = result_date + '月'
                    title = title_str.format(result_date, preps['C'])
                    fig_name = stid + '_' + title.split()[0] + '.png'
                    my_polar(preps, title, RESULTPATH, fig_name)
                    print('{}已形成并保存'.format(fig_name))
                msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "finish", "玫瑰图绘制完成!")
                msg_box.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # 创建窗体对象
    window = MyApp()
    # 窗体显示
    window.show()
    sys.exit(app.exec_())
