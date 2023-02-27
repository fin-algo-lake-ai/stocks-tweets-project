import datetime as dt
import json
import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor, ClearOutputPreprocessor
from nbconvert import HTMLExporter
import os
import shutil
from tqdm import tqdm

# import subprocess

CONF_FILE = 'nb320_config.json'
NB_FILE = 'nb320_added_business_metrics.ipynb'
IS_FAST_CHECK = False


def launch_nb(launch_label: str, iter_no: int, fnames: list, dt_before: str, dt_after: str, dt_split: str,
              label_gen_strategy: str):
    print("1 - Preparing config file")

    tmp_dict = {
        'LAUNCH_LABEL': launch_label,
        'FNAMES': fnames,
        'DROP_RECORDS_BEFORE_DATE_INCLUSIVE': dt_before,
        'DROP_RECORDS_AFTER_DATE_INCLUSIVE': dt_after,
        'OOT_SPLIT_TRAIN_DATE_INCLUSIVE': dt_split,
        'LABEL_GEN_STRATEGY': label_gen_strategy,
        'IS_FAST_CHECK': IS_FAST_CHECK,
    }

    # Save the dictionary to a JSON config file
    with open(CONF_FILE, "w") as f:
        json.dump(tmp_dict, f, indent=4)

    print("2 - Launching the notebook")

    # Method1 - does not allow exceptions in Notebooks
    # subprocess.run(["jupyter", "nbconvert", "--to", "html", "--execute",
    # "--allow-errors", f"--output={out_file}", NB_FILE])

    # Method2
    #
    # https://stackoverflow.com/a/65502489
    with open(NB_FILE, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    cp = ClearOutputPreprocessor()
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')

    err_suffix = ''
    try:
        # Clear the output before execution
        nb, _ = cp.preprocess(nb=nb, resources={})

        # ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        ep.preprocess(nb)
    except CellExecutionError as e:
        if "KeyboardInterrupt" in str(e):
            print("KeyboardInterrupt command in notebook detected -> OK")
        else:
            print(f"Unexpected exception occurred! HTML file will be marked "
                  f"with 'ERROR_AT_ITER=x' suffix, and the current config file is preserved. Details: {e}")
            err_suffix = f"_ERROR_AT_ITER={iter_no}"
            # raise   # Some other error -> promote it

    print("3 - Exporting to HTML")

    # export to html
    html_exporter = HTMLExporter()
    # html_exporter.exclude_input = True
    html_data, resources = html_exporter.from_notebook_node(nb)

    # write to output file
    ts = dt.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    out_file = f"_OUT/{ts}_nb320_added_business_metrics{err_suffix}.html"
    assert not os.path.exists(out_file)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_data)

    # if an error has occurred then the config file is also saved
    if err_suffix:
        out_file = f"_OUT/{ts}_nb320_added_business_metrics_config{err_suffix}.json"
        shutil.copyfile(CONF_FILE, out_file)


def main():
    FILE_NAMES = (
        ['NFLX_RmSW=0_RmRep=0_1y.csv.gz', ],
        ['AMZN_RmSW=0_RmRep=0_1y.csv.gz', ],
        ['AAPL_RmSW=0_RmRep=0_1y.csv.gz', ],
        ['AAPL_RmSW=0_RmRep=0_1y.csv.gz', 'AMZN_RmSW=0_RmRep=0_1y.csv.gz', 'NFLX_RmSW=0_RmRep=0_1y.csv.gz', ]
    )

    LABEL_GEN_STRATEGIES = (
        "d1_C=d1_O=0.5%=2cls",
        #"d2_O=d1_O=0.5%=2cls",
        #"d3_O=d1_O=0.5%=2cls",
        #"d7_O=d1_O=0.5%=2cls",
    )

    DATES = (
        # 6+6
        ['2018-07-20', '2019-07-20', '2019-01-22'],
        ['2018-10-20', '2019-10-20', '2019-04-22'],
        ['2019-01-20', '2020-01-20', '2019-07-22'],
        ['2019-04-20', '2020-04-20', '2019-10-22'],
        ['2019-07-20', '2020-07-20', '2020-01-22'],
        # 3+6
        ['2018-10-20', '2019-07-20', '2019-01-22'],
        ['2019-01-20', '2019-10-20', '2019-04-22'],
        ['2019-04-20', '2020-01-20', '2019-07-22'],
        ['2019-07-20', '2020-04-20', '2019-10-22'],
        ['2019-10-20', '2020-07-20', '2020-01-22'],
        # 9+6
        ['2018-04-20', '2019-07-20', '2019-01-22'],
        ['2018-07-20', '2019-10-20', '2019-04-22'],
        ['2018-10-20', '2020-01-20', '2019-07-22'],
        ['2019-01-20', '2020-04-20', '2019-10-22'],
        ['2019-04-20', '2020-07-20', '2020-01-22'],
        # 12+6
        ['2018-01-20', '2019-07-20', '2019-01-22'],
        ['2018-04-20', '2019-10-20', '2019-04-22'],
        ['2018-07-20', '2020-01-20', '2019-07-22'],
        ['2018-10-20', '2020-04-20', '2019-10-22'],
        ['2019-01-20', '2020-07-20', '2020-01-22'],
        # 15+6
        ['2017-10-20', '2019-07-20', '2019-01-22'],
        ['2018-01-20', '2019-10-20', '2019-04-22'],
        ['2018-04-20', '2020-01-20', '2019-07-22'],
        ['2018-07-20', '2020-04-20', '2019-10-22'],
        ['2018-10-20', '2020-07-20', '2020-01-22'],
    )

    total_iters = len(FILE_NAMES) * len(LABEL_GEN_STRATEGIES) * len(DATES)
    iter_no = 0
    with tqdm(total=total_iters) as pbar:
        for fnames in FILE_NAMES:
            for label_gen_strategy in LABEL_GEN_STRATEGIES:
                for dt_before, dt_after, dt_split in DATES:
                    # Update the progress bar
                    pbar.update()

                    # Check iter no
                    iter_no += 1
                    if iter_no <= 79:
                        print(f"Skip iter {iter_no}")
                        continue

                    # Do the launch
                    launch_label_ = f"Batch launch: d1; FULL; iter={iter_no} of {total_iters}"
                    assert "," not in launch_label_, "The commas will break splitting to columns logic"
                    launch_nb(launch_label_, iter_no, fnames, dt_before, dt_after, dt_split, label_gen_strategy)

    print("Finished.")


if __name__ == '__main__':
    main()
