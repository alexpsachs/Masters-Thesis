"""
The purpose of this script is to merely combine
/home/alex/Thesis/Experiment/Analysis/SYMLOG_metrics/reorient/SYMLOG_metrics_original.ods
and
/home/alex/Thesis/Experiment/Analysis/SYMLOG_metrics/reorient/SYMLOG_metrics_test.ods
into
/home/alex/Thesis/Experiment/Analysis/SYMLOG_metrics/reorient/SYMLOG_metrics.ods
This is done so that the rest of the scripts can remain unchanged for final use, but I can take this shortcut now
"""
import os
from pyexcel_ods3 import get_data, save_data
import code

HOME = os.path.abspath(os.path.join(__file__,'../../../'))
SYM = os.path.join(HOME,'Experiment','Analysis','SYMLOG_metrics')
def helper(exp_name):
    orig_path = os.path.join(SYM,exp_name,'SYMLOG_metrics_original.ods')
    test_path = os.path.join(SYM,exp_name,'SYMLOG_metrics_test.ods')
    out_path = os.path.join(SYM,exp_name,'SYMLOG_metrics.ods')

    orig_data = get_data(orig_path)['SYMLOG']
    # print('orig_path',orig_path)
    # code.interact(local=dict(globals(),**locals()))
    test_data = get_data(test_path)['SYMLOG']

    # code.interact(local=dict(globals(),**locals()))
    all_data = orig_data[0:1] + [row for row in orig_data + test_data if row[0] != 'repository']
    save_data(out_path, {'SYMLOG':all_data})

if __name__ == '__main__':
    helper('old')
    helper('reorient')
    print('done')
