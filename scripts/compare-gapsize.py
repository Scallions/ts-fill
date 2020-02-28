'''
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-02-28 20:05:37
@FilePath     : /gps-ts/scripts/compare-gapsize.py
@Description  : gap size compare
'''




if __name__ == "__main__":
    """genrel code for compare
    """
    result = {}    # result record different between gap sizes and filler functions
    tss = get_tss()
    gap_sizes = [1,2,3,4,5]
    fillers = [1,1,2,3]
    for gap_size in gap_sizes:
        for filler in fillers:
            tss_c = list(map(filler, tss))
            status = get_status_between(tss, tss_c)
            result[gap_size][filler.name] = status