"""验证自编分析程序只能使用公开的 JobContext 成员。"""

from __future__ import annotations

import ast

from minichem_mcp_tools.execution_models import AnalysisJobRequest
from minichem_mcp_tools.open_execution import _analysis_reliability_findings


def test_analysis_preflight_rejects_unknown_job_context_attribute() -> None:
    """不存在的 job_dir 应在提交前给出确定性错误。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "output_path = ctx.job_dir / 'outputs' / 'initial.xyz'\n"
    )

    findings = _analysis_reliability_findings(tree)

    assert findings == [
        {
            "line": 3,
            "kind": "unknown_job_context_attribute",
            "evidence": "ctx.job_dir",
            "message": (
                "JobContext 没有公开成员 'job_dir'；"
                "请使用已声明输出对应的 ctx.output(name)"
            ),
        }
    ]

def test_analysis_preflight_tracks_job_context_import_and_variable_aliases() -> None:
    """JobContext 的导入别名和上下文变量别名都应继续受公开成员检查。"""

    tree = ast.parse(
        "from researchchem_job import JobContext as JC\n"
        "loaded = JC.load()\n"
        "ctx = loaded\n"
        "alias = ctx\n"
        "output_path = alias.job_dir / 'outputs' / 'initial.xyz'\n"
    )

    findings = _analysis_reliability_findings(tree)

    assert findings == [
        {
            "line": 5,
            "kind": "unknown_job_context_attribute",
            "evidence": "alias.job_dir",
            "message": (
                "JobContext 没有公开成员 'job_dir'；"
                "请使用已声明输出对应的 ctx.output(name)"
            ),
        }
    ]


def test_analysis_preflight_accepts_public_job_context_members() -> None:
    """公开输入输出助手和目录属性不应被误报。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "source = ctx.input('source')\n"
        "target = ctx.output('target')\n"
        "target.write_text(source.read_text())\n"
        "ctx.register_output('target')\n"
        "dirs = (ctx.root, ctx.inputs_dir, ctx.outputs_dir, ctx.report_dir, ctx.logs_dir)\n"
    )

    assert _analysis_reliability_findings(tree) == []


def test_analysis_preflight_rejects_registered_output_written_to_logs() -> None:
    """默认注册前写到日志目录会确定性错过声明产物路径。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "target = ctx.logs_dir / 'initial.xyz'\n"
        "target.write_text('1\\nexample\\nH 0 0 0\\n')\n"
        "ctx.register_output('initial_xyz')\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/generate_xyz.py",
        outputs=[
            {
                "name": "initial_xyz",
                "path": "outputs/initial.xyz",
                "semantic_type": "XYZFile",
                "media_type": "chemical/x-xyz",
            }
        ],
    )

    findings = _analysis_reliability_findings(tree, request=request)

    assert findings == [
        {
            "line": 5,
            "kind": "declared_output_path_mismatch",
            "evidence": "logs/initial.xyz != outputs/initial.xyz",
            "message": (
                "声明输出 'initial_xyz' 必须写入 'outputs/initial.xyz'；"
                "程序实际写入 'logs/initial.xyz' 后按默认路径注册"
            ),
        }
    ]


def test_analysis_preflight_accepts_writes_to_declared_output_path() -> None:
    """公开 helper 或逐字匹配的日志声明仍保持可执行。"""

    helper_tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "target = ctx.output('initial_xyz')\n"
        "target.write_text('1\\nexample\\nH 0 0 0\\n')\n"
        "ctx.register_output('initial_xyz')\n"
    )
    output_request = AnalysisJobRequest(
        runtime="core",
        script_path="code/generate_xyz.py",
        outputs=[
            {
                "name": "initial_xyz",
                "path": "outputs/initial.xyz",
                "semantic_type": "XYZFile",
            }
        ],
    )
    logs_tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "target = ctx.logs_dir / 'diagnostic.txt'\n"
        "target.write_text('ok')\n"
        "ctx.register_output('diagnostic')\n"
    )
    logs_request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_diagnostic.py",
        outputs=[
            {
                "name": "diagnostic",
                "path": "logs/diagnostic.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    assert _analysis_reliability_findings(helper_tree, request=output_request) == []
    assert _analysis_reliability_findings(logs_tree, request=logs_request) == []


def test_analysis_preflight_rejects_explicit_wrong_registration_path() -> None:
    """即使声明路径已有文件，显式注册另一路径仍会在运行时失败。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "declared = ctx.output('result')\n"
        "declared.write_text('ok')\n"
        "wrong = ctx.logs_dir / 'result.txt'\n"
        "wrong.write_text('wrong')\n"
        "ctx.register_output('result', wrong)\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_result.py",
        outputs=[
            {
                "name": "result",
                "path": "outputs/result.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    findings = _analysis_reliability_findings(tree, request=request)

    assert findings == [
        {
            "line": 7,
            "kind": "declared_output_path_mismatch",
            "evidence": "logs/result.txt != outputs/result.txt",
            "message": (
                "声明输出 'result' 必须写入 'outputs/result.txt'；"
                "程序显式注册了 'logs/result.txt'"
            ),
        }
    ]


def test_analysis_preflight_rejects_literal_wrong_registration_path() -> None:
    """register_output 的逐字相对路径必须与声明路径完全一致。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "ctx.register_output('result', 'logs/result.txt')\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_result.py",
        outputs=[
            {
                "name": "result",
                "path": "outputs/result.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    findings = _analysis_reliability_findings(tree, request=request)

    assert findings == [
        {
            "line": 3,
            "kind": "declared_output_path_mismatch",
            "evidence": "logs/result.txt != outputs/result.txt",
            "message": (
                "声明输出 'result' 必须写入 'outputs/result.txt'；"
                "程序显式注册了 'logs/result.txt'"
            ),
        }
    ]


def test_analysis_preflight_rejects_absolute_registration_path() -> None:
    """绝对输出路径会越出任务根目录，必须在提交前拒绝。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "ctx.register_output('result', '/tmp/result.txt')\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_result.py",
        outputs=[
            {
                "name": "result",
                "path": "outputs/result.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    findings = _analysis_reliability_findings(tree, request=request)

    assert findings == [
        {
            "line": 3,
            "kind": "declared_output_path_mismatch",
            "evidence": "/tmp/result.txt != outputs/result.txt",
            "message": (
                "声明输出 'result' 必须写入 'outputs/result.txt'；"
                "程序显式注册了 '/tmp/result.txt'"
            ),
        }
    ]


def test_analysis_preflight_does_not_guess_dynamic_registration_path() -> None:
    """动态路径无法静态确定时不得臆测为错误。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "path = choose_path()\n"
        "ctx.register_output('result', path)\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_result.py",
        outputs=[
            {
                "name": "result",
                "path": "outputs/result.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    assert _analysis_reliability_findings(tree, request=request) == []


def test_analysis_preflight_accepts_literal_logs_diagnostic_registration() -> None:
    """日志诊断输出按其声明的 logs 路径注册时保持可用。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "ctx.register_output('diagnostic', 'logs/diagnostic.txt')\n"
    )
    request = AnalysisJobRequest(
        runtime="core",
        script_path="code/write_diagnostic.py",
        outputs=[
            {
                "name": "diagnostic",
                "path": "logs/diagnostic.txt",
                "semantic_type": "DiagnosticText",
            }
        ],
    )

    assert _analysis_reliability_findings(tree, request=request) == []


def test_analysis_preflight_rejects_output_path_attribute() -> None:
    """JobContext.output(name) 已返回 pathlib.Path，不应再访问 .path。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "output_path = ctx.output('result').path\n"
    )

    findings = _analysis_reliability_findings(tree)

    assert findings == [
        {
            "line": 3,
            "kind": "job_context_output_path_attribute",
            "evidence": "JobContext.output(...).path",
            "message": (
                "JobContext.output(name) already returns pathlib.Path; the Path object "
                "has no .path attribute"
            ),
        }
    ]


def test_analysis_preflight_rejects_input_path_attribute() -> None:
    """JobContext.input(name) 已返回 pathlib.Path，不应再访问 .path。"""

    tree = ast.parse(
        "from researchchem_job import JobContext\n"
        "ctx = JobContext.load()\n"
        "input_path = ctx.input('source').path\n"
    )

    findings = _analysis_reliability_findings(tree)

    assert findings == [
        {
            "line": 3,
            "kind": "job_context_input_path_attribute",
            "evidence": "JobContext.input(...).path",
            "message": (
                "JobContext.input(name) already returns pathlib.Path; the Path object "
                "has no .path attribute"
            ),
        }
    ]
