from pathlib import Path
import json
from src.utils.runs import create_run,write_txt_report


def config(arch='mlp'):
    return {'architecture':{'name':arch},'experiment':{'seed':42},
            'dataset':{'path':'datasets/raw/example.csv'},
            'embedding':{'provider':'npy','path':'data.npy'},
            'report':{'primary_metric':'macro_f1'}}


def test_timestamped_runs_do_not_overwrite(tmp_path):
    a,da=create_run(tmp_path,'experiment')
    b,db=create_run(tmp_path,'experiment')
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert da['models'].is_dir() and da['reports'].is_dir()
    assert db['figures'].is_dir()


def test_report_highlights_primary_metric_and_retains_files(tmp_path):
    run,dirs=create_run(tmp_path,'mlp')
    report=write_txt_report(dirs,config(),{'test':{'accuracy':.92,'macro_f1':.81}},run)
    assert report.exists()
    content=report.read_text(encoding='utf-8')
    assert 'macro_f1: 0.810000' in content and 'METRICA PRINCIPAL' in content
    assert 'accuracy: 0.920000' in content
    try:
        write_txt_report(dirs,config(),{'test':{'accuracy':.1,'macro_f1':.1}},run)
        assert False, 'Should refuse to overwrite report'
    except FileExistsError:
        pass
