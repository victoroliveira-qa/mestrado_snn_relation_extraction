from src.utils.output import output_dirs

def test_output_dirs(tmp_path):
    dirs=output_dirs(tmp_path/'run')
    assert all(path.is_dir() for path in dirs.values())
    assert dirs['models'] != dirs['metrics']
