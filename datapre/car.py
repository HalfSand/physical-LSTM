import pandas as pd
import json
import os
from trajnetplusplustools import TrackRow


class Car:
    def __init__(self):
        self.scale_factors = None
        self.temp = None

    def calculate_scale_factors(self, df):
        # 提取你需要的列
        columns = ['x', 'y', 'xVelocity', 'yVelocity']

        # 找到每列的最大值并除以100
        #self.scale_factors = [df[col].abs().max() / 100 for col in columns]
        # self.scale_factors = [4, 0.3, 0.5, 0.015]
        self.scale_factors = [2, 0.5, 0.01, 0.02]

    def read_line(self, line):
        line = [e for e in line.split('\t') if e != '']
        if len(line) != 6:
            print(f"Unexpected line format: {line}")
            return None  # 或者处理错误的方式
        return TrackRow(
            int(float(line[0])),
            int(float(line[1])),
            float(line[2]),
            float(line[3]),
            float(line[4]),
            float(line[5])
        )

    def read_csv(self, sc, input_file):
        print('processing ' + input_file)

        # 使用 pandas 读取 csv 文件
        df = pd.read_csv(input_file)

        # 选取你需要的列
        # df = df[['frame', 'id', 'x', 'y', 'xVelocity', 'yVelocity']]

        # 定义一个列名映射字典
        column_mapping = {
            'frame': 'frame',
            'id': 'id',
            'x': 'x',
            'y': 'y',
            'xVelocity': 'xVelocity',
            'yVelocity': 'yVelocity'
        }

        # 使用字典来选取和重命名列
        df = df[list(column_mapping.keys())].rename(columns=column_mapping)
        # 计算缩放因子
        self.calculate_scale_factors(df)

        # 应用缩放因子
        for i, col in enumerate(['x', 'y', 'xVelocity', 'yVelocity']):
            df[col] = df[col] / self.scale_factors[i]

        # 获取输入文件的路径和文件名，用于生成输出文件的路径
        dir_path, file_name = os.path.split(input_file)
        base_name, _ = os.path.splitext(file_name)
        self.temp = os.path.join(dir_path, base_name + '.txt')

        # 将 DataFrame 对象写入 txt 文件，不包含列名和索引，每列的值之间用 '\t' 分隔
        df.to_csv(self.temp, sep='\t', header=False, index=False)

        return (sc
                .textFile(self.temp)
                .map(self.read_line)
                .cache())

    def delete_temp(self):
        if self.temp is not None and os.path.exists(self.temp):
            os.remove(self.temp)
            print(f"Temp file {self.temp} has been deleted.")
        else:
            print("No temp file to delete.")

    def append_scaler(self, ndjson_file):
        if self.scale_factors is not None:
            scale_dict = {'scaler':
                              {'x': self.scale_factors[0],
                               'y': self.scale_factors[1],
                               'xVelocity': self.scale_factors[2],
                               'yVelocity': self.scale_factors[3]}}
            with open(ndjson_file, 'a') as f:
                f.write(json.dumps(scale_dict) + '\n')
        else:
            print("scale_factors is None.")
