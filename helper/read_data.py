# 该文件实现从表格文件中提取所需的加工时间矩阵和换模时间矩阵
import pandas as pd


def get_info_from_file(filename, selected_job=None, print_table=False):
    """
    从excel表格中读取加工时间和换模时间，并添加工件0和工序0相关数据。

    :param filename: excel表格文件名。
    :param selected_job: 可以传入list，指定要考虑的job（所有jobs的子集）。
    :return: 两个二维list，分别表示加工时间和换模时间信息。
    """
    processing_time = pd.read_excel(filename, sheet_name="加工时间", index_col=0)
    setup_time = pd.read_excel(filename, sheet_name="换模时间", index_col=0)
    processing_time = processing_time.reindex(index=['M0'] + list(processing_time.index),
                                              columns=['J0'] + list(processing_time.columns),
                                              fill_value=0)
    setup_time = setup_time.reindex(index=['J0'] + list(setup_time.index),
                                              columns=['J0'] + list(setup_time.columns),
                                              fill_value=0)
    if selected_job is not None:
        processing_time = processing_time.iloc[:, [0] + selected_job]
        setup_time = setup_time.iloc[[0] + selected_job, [0] + selected_job]

    if print_table:
        print('加工时间表')
        print(processing_time)
        print()
        print('换模时间表')
        print(setup_time)
    return processing_time.values.tolist(), setup_time.values.tolist()


if __name__ == '__main__':
    p, s = get_info_from_file('../data/case 2.xlsx', print_table=True)
    # all_s = []
    # for line in s:
    #     all_s.extend(line)
    # all_s.sort()
    # print(all_s)
    #
    # seq = [11, 16, 5, 8, 3, 2, 10, 6, 1, 13, 14, 15, 12, 7, 9, 4]
    # for i in range(16-1):
    #     print(s[seq[i]][seq[i+1]])
