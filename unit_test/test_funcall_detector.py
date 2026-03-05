from funcall_detector import detect_funcall


def test_show_procedure_panel_hit():
    result = detect_funcall("现在第几步了")
    assert result is not None
    assert result.command == "ShowProcedurePanel"


def test_current_experiment_operation_hit():
    result = detect_funcall("这一步怎么操作")
    assert result is not None
    assert result.command == "CurrentExperimentOperation"


def test_switch_experiment_scene_hit():
    result = detect_funcall("切换到PCR实验")
    assert result is not None
    assert result.command == "SwitchExperimentScene"


def test_excluded_by_condition_clause():
    result = detect_funcall("如果离心机出了问题这步怎么办")
    assert result is None


def test_excluded_by_long_technical_sentence():
    result = detect_funcall("加完样之后下一步怎么做")
    assert result is None

