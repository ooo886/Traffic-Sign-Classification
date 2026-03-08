import pandas as pd
import re


def parse_class_data(text_content, dataset_name):
    """
    解析文本内容并转换为DataFrame
    """
    data = []
    lines = text_content.strip().split('\n')

    for line in lines:
        # 使用正则表达式提取类别和数量
        # 匹配格式如 "0: 118张" 或 "9：0张"（注意全角冒号）
        match = re.search(r'(\d+)[:：]\s*(\d+)', line)
        if match:
            class_num = int(match.group(1))
            count = int(match.group(2))
            data.append({'类别': class_num, dataset_name: count})

    return pd.DataFrame(data)


# 你的数据
data_text = """0: 118张     
1: 40张
2: 80张
3: 260张
4: 98张
5: 194张
6: 78张
7: 152张
8: 8张
9: 2张
10: 70张
11: 138张
12: 96张
13: 36张
14: 128张
15: 22张
16: 142张
17: 130张
18: 8张
19: 4张
20: 18张
21: 12张
22: 18张
23: 14张
24: 100张
25: 2张
26: 126张
27: 28张
28: 446张
29: 44张
30: 150张
31: 42张
32: 14张
33: 4张
34: 26张
35: 156张
36: 40张
37: 58张
38: 30张
39: 34张
40: 32张
41: 18张
42: 32张
43: 82张
44: 30张
45: 24张
46: 18张
47: 12张
48: 10张
49: 42张
50: 56张
51: 8张
52: 36张
53: 2张
54: 324张
55: 162张
56: 110张
57: 6张"""

test_text = """0: 7张
1: 5张
2: 30张
3: 46张
4: 27张
5: 22张
6: 17张
7: 25张
8: 7张
9：0张
10: 22张
11: 59张
12: 12张
13: 41张
14: 4张
15: 16张
16: 42张
17: 40张
18：0张
19：0张
20: 1张
21: 4张
22: 2张
23: 6张
24: 14张
25: 1张
26: 70张
27: 13张
28: 42张
29: 18张
30: 16张
31: 12张
32: 2张
33：0张
34: 5张
35: 23张
36: 6张
37: 11张
38: 18张
39: 15张
40: 1张
41: 4张
42: 11张
43: 58张
44: 11张
45: 1张
46: 4张
47: 6张
48: 3张
49: 31张
50: 12张
51: 1张
52: 17张
53: 1张
54: 86张
55: 25张
56: 23张
57: 1张"""

val_text = """0: 7张
1: 7张
2: 30张
3: 38张
4: 31张
5: 28张
6: 13张
7: 25张
8: 7张
9：0张
10: 38张
11: 71张
12: 10张
13: 51张
14: 8张
15: 20张
16: 34张
17: 44张
18：0张
19：0张
20: 1张
21: 8张
22: 6张
23: 4张
24: 12张
25: 1张
26: 64张
27: 11张
28: 26张
29: 8张
30: 18张
31: 6张
32：0张
33：0张
34: 3张
35: 23张
36: 6张
37: 15张
38: 22张
39: 15张
40: 7张
41: 4张
42: 7张
43: 58张
44: 13张
45: 1张
46: 10张
47: 4张
48: 3张
49: 11张
50: 8张
51: 3张
52: 13张
53: 1张
54: 90张
55: 33张
56: 17张
57: 3张"""

# 解析三个数据集
df_data = parse_class_data(data_text, 'DATA')
df_test = parse_class_data(test_text, 'TEST')
df_val = parse_class_data(val_text, 'VAL')

# 合并所有数据
df_merged = pd.merge(df_data, df_test, on='类别', how='outer')
df_merged = pd.merge(df_merged, df_val, on='类别', how='outer')

# 按类别排序
df_merged = df_merged.sort_values('类别').reset_index(drop=True)

# 填充NaN为0
df_merged = df_merged.fillna(0)

# 将浮点数转换为整数
df_merged['DATA'] = df_merged['DATA'].astype(int)
df_merged['TEST'] = df_merged['TEST'].astype(int)
df_merged['VAL'] = df_merged['VAL'].astype(int)

# 添加总计列
df_merged['总计'] = df_merged['DATA'] + df_merged['TEST'] + df_merged['VAL']

# 保存为Excel
excel_path = r"C:\Users\Administrator\Desktop\traffic_sign_dataset_statistics.xlsx"
df_merged.to_excel(excel_path, index=False)

print(f"✅ Excel文件已保存到: {excel_path}")
print("\n📊 数据预览:")
print(df_merged.head(10))