"""Investigation and Experiment 批次的确定性协调模块。

本模块在一个小型主循环中集中实现并行调查、逐项持久化、完整五项 Review 屏障和领域路由
公开入口中。模型、MiniChem 和下一批假设生成均通过调用方注入；协调器只接收
已经完成的领域结果，并负责验证归属、推进纯领域状态和按 CAS 版本保存快照。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

from .domain import (
    ApplyBatchValidation,
    BatchValidation,
    InvestigationResult,
    InvestigationSession,
    InvestigationState,
    RecordInvestigationResult,
    apply_investigation_event,
)
from .sqlite_store import (
    InvestigationRevisionConflict,
    InvestigationSQLiteStore,
    InvestigationStateSnapshot,
)


InvestigationRoute: TypeAlias = Literal[
    "investigation",
    "waiting_investigation_job",
    "repair_required",
    "accept",
    "answerability_explanation",
    "regenerate_hypotheses",
    "failed",
]
InitialInvestigator: TypeAlias = Callable[[InvestigationSession], InvestigationResult]
RepairInvestigator: TypeAlias = Callable[
    [InvestigationSession, str], InvestigationResult
]
BatchReviewer: TypeAlias = Callable[
    [
        InvestigationState,
        tuple[InvestigationResult, ...],
        tuple[tuple[str, str], ...],
    ],
    BatchValidation,
]

_TERMINAL_ROUTES = frozenset(
    {
        "accept",
        "answerability_explanation",
        "regenerate_hypotheses",
        "failed",
    }
)
_REVIEWABLE_PHASES = frozenset(
    {
        "ready_for_batch_validation",
        "validated",
        "marked_for_hypothesis_revision",
    }
)


class InvestigationCoordinatorConflict(InvestigationRevisionConflict):
    """协调器无法安全合并并发事实时报告的明确 CAS 冲突。

    外部调用方可以按既有 ``InvestigationRevisionConflict`` 捕获此异常。协调器
    只有在恢复快照中找不到本次相同结果或验证记录时才抛出它，不会覆盖另一进程
    已经保存的事实，也不会自动重新调用模型。
    """

    def __init__(
        self,
        cause: InvestigationRevisionConflict,
        *,
        operation: str,
    ) -> None:
        self.operation = operation
        super().__init__(
            cause.task_id,
            cause.expected_revision,
            cause.actual_revision,
        )
        self.args = (
            f"{operation} 的调查状态 CAS 冲突：task_id={cause.task_id}，"
            f"expected_revision={cause.expected_revision}，"
            f"actual_revision={cause.actual_revision}；"
            "恢复快照没有发现可重放的相同事实",
        )


@dataclass(frozen=True)
class InvestigationBatchOutcome:
    """一次协调入口返回的最小冻结审计结果。

    ``state`` 和 ``revision`` 是同一份最终快照，``route`` 是领域层决定的下一步。
    ``review_called`` 表示本次入口是否真正调用了批次 Review；调用计数只统计本次
    入口，``replay_count`` 统计已保存结果、验证或 CAS 重放，不把它们误报为模型调用。
    """

    state: InvestigationState
    revision: int
    route: InvestigationRoute
    review_called: bool
    initial_investigator_calls: int
    repair_investigator_calls: int
    review_calls: int
    replay_count: int


@dataclass
class _CallCounters:
    """协调入口内部使用的可变调用计数；不暴露给外部。"""

    initial_investigator_calls: int = 0
    repair_investigator_calls: int = 0
    review_calls: int = 0
    replay_count: int = 0


def _load_or_save_initial_state(
    store: InvestigationSQLiteStore,
    initial_state: InvestigationState | None,
    task_id: str,
) -> InvestigationStateSnapshot:
    """恢复任务快照；首次运行时按 revision=0 保存调用方提供的初态。

    已有快照始终优先，避免恢复入口用陈旧的初态覆盖运行中的结果。首次 CAS 创建
    如果与另一进程竞争，只有完整初态相同才视为幂等；其它内容直接报告冲突。
    """

    snapshot = store.load_investigation_snapshot(task_id)
    if snapshot is not None:
        return snapshot
    if initial_state is None:
        raise ValueError("任务尚无持久化状态，首次运行必须提供 initial_state")
    if initial_state.task_id != task_id:
        raise ValueError("initial_state.task_id 与 task_id 不匹配")
    try:
        revision = store.save_investigation_state(initial_state, expected_revision=0)
    except InvestigationRevisionConflict as exc:
        competing = store.load_investigation_snapshot(task_id)
        if competing is not None and competing.state == initial_state:
            return competing
        raise InvestigationCoordinatorConflict(exc, operation="保存初始状态") from exc
    return InvestigationStateSnapshot(state=initial_state, revision=revision)


def _repair_advice_for_session(
    state: InvestigationState,
    session: InvestigationSession,
) -> str:
    """从当前批次最近一次 Review 中取出指定组合的返修建议。"""

    for validation in reversed(state.validations):
        if validation.batch_id != state.batch_id:
            continue
        for finding in validation.findings:
            if finding.combination_id == session.combination_id:
                if finding.validity != "fail" or finding.repair_advice is None:
                    break
                return finding.repair_advice
    raise ValueError(
        f"组合 {session.combination_id} 处于 repair_pending，但没有可用的返修建议"
    )


def _select_investigation_targets(
    state: InvestigationState,
) -> tuple[tuple[int, InvestigationSession, str, str | None], ...]:
    """按 state.sessions 顺序选择本轮唯一可调用的首次调查或返修会话。"""

    targets: list[tuple[int, InvestigationSession, str, str | None]] = []
    for session in state.sessions:
        if session.phase == "created":
            targets.append((len(targets), session, "initial", None))
        elif session.phase == "repair_pending":
            targets.append(
                (
                    len(targets),
                    session,
                    "repair",
                    _repair_advice_for_session(state, session),
                )
            )
    return tuple(targets)


def _validate_investigation_result_for_session(
    state: InvestigationState,
    session: InvestigationSession,
    result: InvestigationResult,
) -> None:
    """核对外部结果的任务、组合、会话线程、scope 和版本归属。"""

    if not isinstance(result, InvestigationResult):
        raise TypeError("调查器必须返回 InvestigationResult")
    if result.task_id != state.task_id or result.batch_id != state.batch_id:
        raise ValueError("InvestigationResult 的 task_id/batch_id 与当前状态不匹配")
    if result.combination_id != session.combination_id:
        raise ValueError("InvestigationResult 的 combination_id 与调用会话不匹配")
    if (
        session.codex_thread_id is not None
        and result.codex_thread_id != session.codex_thread_id
    ):
        raise ValueError("InvestigationResult 的 Codex thread 与调用会话不匹配")
    if result.scope_id != session.scope_id:
        raise ValueError("InvestigationResult 的 scope 与调用会话不匹配")
    if any(
        saved.combination_id == result.combination_id
        and saved.result_version == result.result_version
        for saved in state.results
    ):
        raise ValueError("同一组合不能复用已经保存的 result_version")


def _persist_investigation_result(
    store: InvestigationSQLiteStore,
    state: InvestigationState,
    revision: int,
    session: InvestigationSession,
    result: InvestigationResult,
) -> tuple[InvestigationState, int, bool]:
    """推进并逐项保存一个结果；同版本外部事实只重放，不重复调用调查器。"""

    _validate_investigation_result_for_session(state, session, result)
    next_state = apply_investigation_event(
        state,
        RecordInvestigationResult(result=result),
    )
    try:
        next_revision = store.save_investigation_state(
            next_state,
            expected_revision=revision,
        )
        # SQLite 对完全相同快照提供幂等返回而不递增 revision；这也是另一协调器
        # 已经先保存同一结果时的重放信号，即使没有抛出 CAS 异常。
        return next_state, next_revision, next_revision != revision + 1
    except InvestigationRevisionConflict as exc:
        competing = store.load_investigation_snapshot(state.task_id)
        saved = None
        if competing is not None:
            saved = next(
                (
                    item
                    for item in competing.state.results
                    if item.combination_id == result.combination_id
                    and item.result_version == result.result_version
                ),
                None,
            )
        if saved is not None:
            if saved != result:
                raise InvestigationCoordinatorConflict(
                    exc,
                    operation="保存调查结果",
                ) from exc
            assert competing is not None  # 由 saved 的存在保证
            return competing.state, competing.revision, True
        raise InvestigationCoordinatorConflict(exc, operation="保存调查结果") from exc


def _current_results_and_vector(
    state: InvestigationState,
) -> tuple[tuple[InvestigationResult, ...], tuple[tuple[str, str], ...]] | None:
    """按固定会话顺序构造完整五项当前结果和版本向量。"""

    if any(session.phase not in _REVIEWABLE_PHASES for session in state.sessions):
        return None
    results_by_key = {
        (result.combination_id, result.result_version): result
        for result in state.results
    }
    current_results: list[InvestigationResult] = []
    vector: list[tuple[str, str]] = []
    for session in state.sessions:
        if session.result_version is None:
            return None
        key = (session.combination_id, session.result_version)
        result = results_by_key.get(key)
        if result is None:
            raise ValueError("会话当前 result_version 缺少对应的已保存调查结果")
        current_results.append(result)
        vector.append(key)
    return tuple(current_results), tuple(vector)


def _validate_batch_review_for_snapshot(
    state: InvestigationState,
    vector: tuple[tuple[str, str], ...],
    validation: BatchValidation,
) -> None:
    """严格核对 Review 的任务、批次、五组合和完整结果版本配对。"""

    if not isinstance(validation, BatchValidation):
        raise TypeError("批次 Review 必须返回 BatchValidation")
    if validation.task_id != state.task_id or validation.batch_id != state.batch_id:
        raise ValueError("BatchValidation 的 task_id/batch_id 与当前状态不匹配")
    expected = dict(vector)
    received = dict(zip(validation.combination_ids, validation.result_versions))
    if set(received) != set(expected) or received != expected:
        raise ValueError("BatchValidation 必须严格绑定当前完整五项结果版本向量")


def _saved_validation_for_key(
    state: InvestigationState,
    validation: BatchValidation,
) -> BatchValidation | None:
    """查找当前快照或历史中同一完整向量的验证记录。"""

    records = list(state.validations)
    for history in state.history:
        records.extend(history.validations)
    for saved in records:
        if saved.idempotency_key == validation.idempotency_key:
            return saved
    return None


def _persist_batch_validation(
    store: InvestigationSQLiteStore,
    state: InvestigationState,
    revision: int,
    validation: BatchValidation,
) -> tuple[InvestigationState, int, bool]:
    """应用领域 Review 路由并与状态快照原子 CAS 保存。"""

    next_state = apply_investigation_event(
        state,
        ApplyBatchValidation(validation=validation),
    )
    try:
        next_revision = store.save_investigation_state(
            next_state,
            expected_revision=revision,
        )
        # 验证和状态是同一快照；旧 revision 下的相同快照返回表示验证已重放。
        return next_state, next_revision, next_revision != revision + 1
    except InvestigationRevisionConflict as exc:
        competing = store.load_investigation_snapshot(state.task_id)
        saved = (
            _saved_validation_for_key(competing.state, validation)
            if competing is not None
            else None
        )
        if saved is not None:
            if saved != validation:
                raise InvestigationCoordinatorConflict(
                    exc,
                    operation="保存批次验证",
                ) from exc
            assert competing is not None  # 由 saved 的存在保证
            return competing.state, competing.revision, True
        raise InvestigationCoordinatorConflict(exc, operation="保存批次验证") from exc


def _validate_investigation_dispatch_inputs(
    targets: tuple[tuple[int, InvestigationSession, str, str | None], ...],
    *,
    initial_investigator: InitialInvestigator | None,
    repair_investigator: RepairInvestigator | None,
    max_workers: int,
) -> int:
    """校验调查分发依赖并返回受目标数量限制的 worker 数量。

    输入是固定顺序的调查目标、首次/返修调查器和并行预算；函数只做合同校验，
    不调用外部调查器或写入状态。返回值是正整数 worker 数量；预算类型、预算范围
    或所需调查器缺失时抛出 TypeError/ValueError，目标为空由上层提前返回。
    """

    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers 必须是正整数")
    if max_workers < 1:
        raise ValueError("max_workers 必须是正整数")
    if any(kind == "initial" and initial_investigator is None for _, _, kind, _ in targets):
        raise ValueError("存在 created 会话，但没有注入 initial_investigator")
    if any(kind == "repair" and repair_investigator is None for _, _, kind, _ in targets):
        raise ValueError("存在 repair_pending 会话，但没有注入 repair_investigator")
    return min(max_workers, len(targets))


def _collect_investigation_dispatch_results(
    targets: tuple[tuple[int, InvestigationSession, str, str | None], ...],
    *,
    initial_investigator: InitialInvestigator | None,
    repair_investigator: RepairInvestigator | None,
    counters: _CallCounters,
    worker_count: int,
) -> tuple[dict[int, InvestigationResult], BaseException | None]:
    """并行收集调查结果并保留首个调查器错误。

    输入是已通过依赖校验的目标和两个调查器；函数只调用外部调查器，不写入
    Investigation 状态。返回按目标索引保存的成功结果和首个异常；后续 future
    仍会被收集，避免线程池退出时丢失其它已完成结果，但不会自动重试或吞掉异常。
    """

    pending_results: dict[int, InvestigationResult] = {}
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[InvestigationResult], tuple[int, InvestigationSession]] = {}
        for index, session, kind, advice in targets:
            if kind == "initial":
                assert initial_investigator is not None
                counters.initial_investigator_calls += 1
                future = executor.submit(initial_investigator, session)
            else:
                assert repair_investigator is not None
                assert advice is not None
                counters.repair_investigator_calls += 1
                future = executor.submit(repair_investigator, session, advice)
            futures[future] = (index, session)

        for future in as_completed(futures):
            index, _ = futures[future]
            try:
                pending_results[index] = future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
    return pending_results, first_error


def _persist_investigation_dispatch_results(
    store: InvestigationSQLiteStore,
    state: InvestigationState,
    revision: int,
    targets: tuple[tuple[int, InvestigationSession, str, str | None], ...],
    pending_results: dict[int, InvestigationResult],
    counters: _CallCounters,
) -> tuple[InvestigationState, int, BaseException | None]:
    """按固定会话顺序验证并保存已完成调查结果。

    输入是同一轮 worker 的成功结果；函数通过领域事件和 CAS 保存逐项推进状态，
    记录可安全重放的结果，继续保存其它成功项。返回最新状态、revision 和首个
    持久化异常；CAS、归属或领域合同错误不会被替换为科学路由。
    """

    persistence_error: BaseException | None = None
    for index, session, _, _ in targets:
        if index not in pending_results:
            continue
        result = pending_results[index]
        try:
            state, revision, replayed = _persist_investigation_result(
                store,
                state,
                revision,
                session,
                result,
            )
            if replayed:
                counters.replay_count += 1
        except BaseException as exc:
            if persistence_error is None:
                persistence_error = exc
    return state, revision, persistence_error


def _invoke_and_persist_investigations(
    store: InvestigationSQLiteStore,
    state: InvestigationState,
    revision: int,
    targets: tuple[tuple[int, InvestigationSession, str, str | None], ...],
    *,
    initial_investigator: InitialInvestigator | None,
    repair_investigator: RepairInvestigator | None,
    counters: _CallCounters,
    max_workers: int,
) -> tuple[InvestigationState, int]:
    """并行调用目标调查器，并按固定会话顺序逐项验证和保存结果。

    输入是一个协调批次选出的目标集合和已注入调查器；函数先校验分发合同，
    再等待全部 future，最后按 state.sessions 顺序写入结果。调查器错误仍会在
    已完成结果落盘后原样抛出；持久化错误也不会转成科学路由或自动重试。
    返回最新状态和 CAS revision；空目标直接返回原状态。
    """

    if not targets:
        return state, revision
    worker_count = _validate_investigation_dispatch_inputs(
        targets,
        initial_investigator=initial_investigator,
        repair_investigator=repair_investigator,
        max_workers=max_workers,
    )
    pending_results, first_error = _collect_investigation_dispatch_results(
        targets,
        initial_investigator=initial_investigator,
        repair_investigator=repair_investigator,
        counters=counters,
        worker_count=worker_count,
    )
    state, revision, persistence_error = _persist_investigation_dispatch_results(
        store,
        state,
        revision,
        targets,
        pending_results,
        counters,
    )
    if first_error is not None:
        raise first_error
    if persistence_error is not None:
        raise persistence_error
    return state, revision




def _make_outcome(
    state: InvestigationState,
    revision: int,
    counters: _CallCounters,
) -> InvestigationBatchOutcome:
    """从最终快照和内部计数构造冻结返回值。"""

    return InvestigationBatchOutcome(
        state=state,
        revision=revision,
        route=state.route,
        review_called=counters.review_calls > 0,
        initial_investigator_calls=counters.initial_investigator_calls,
        repair_investigator_calls=counters.repair_investigator_calls,
        review_calls=counters.review_calls,
        replay_count=counters.replay_count,
    )


def run_investigation_and_experiment_batch(
    store: InvestigationSQLiteStore,
    initial_state: InvestigationState | None = None,
    initial_investigator: InitialInvestigator | None = None,
    repair_investigator: RepairInvestigator | None = None,
    batch_reviewer: BatchReviewer | None = None,
    *,
    task_id: str | None = None,
    max_workers: int = 5,
) -> InvestigationBatchOutcome:
    """恢复并完成一个 Investigation and Experiment 批次。

    输入是 SQLite store、首次运行时的五会话 ``initial_state``，以及三个注入的
    外部副作用调用：首次调查器、带 Review 建议的返修调查器和接收完整五项当前
    结果/版本向量的批次 Review。动作会先恢复或保存初态，再最多五路并行调用需要
    的调查器；每个成功结果都经领域入口推进后单独 CAS 落盘，屏障满足后先查询
    ``read_saved_batch_validation``，未命中才调用 Review。返回值冻结最终 state、
    revision、领域 route 和本次调用/重放计数。

    调查器或 Review 的异常、领域合同错误、数据损坏和无法安全合并的 CAS 冲突都
    直接失败；已成功保存的部分会在下一次入口调用时恢复。协调器不能解决外部
    MiniChem/模型已经产生副作用但尚未保存 receipt 的 exactly-once 问题：恢复时只
    能依据快照再次注入尚未落盘会话，具体是否重发及如何用 operation receipt 去重
    必须由外部运行时负责。
    """

    if task_id is None:
        if initial_state is None:
            raise ValueError("首次入口需要 initial_state 或明确的 task_id")
        task_id = initial_state.task_id
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id 不能为空")
    if initial_state is not None and initial_state.task_id != task_id:
        raise ValueError("initial_state.task_id 与 task_id 不匹配")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers 必须是正整数")
    if max_workers < 1:
        raise ValueError("max_workers 必须是正整数")

    snapshot = _load_or_save_initial_state(store, initial_state, task_id)
    state = snapshot.state
    revision = snapshot.revision
    counters = _CallCounters()

    while True:
        if state.route in _TERMINAL_ROUTES:
            return _make_outcome(state, revision, counters)

        targets = _select_investigation_targets(state)
        if targets:
            state, revision = _invoke_and_persist_investigations(
                store,
                state,
                revision,
                targets,
                initial_investigator=initial_investigator,
                repair_investigator=repair_investigator,
                counters=counters,
                max_workers=max_workers,
            )
            # 调查结果保存完后进入同一轮的完整五项 Review 屏障。

        review_input = _current_results_and_vector(state)
        if review_input is not None:
            current_results, vector = review_input
            validation = store.read_saved_batch_validation(
                state.task_id,
                state.batch_id,
                vector,
            )
            if validation is None:
                if batch_reviewer is None:
                    raise ValueError("五项结果已齐备，但没有注入 batch_reviewer")
                validation = batch_reviewer(state, current_results, vector)
                counters.review_calls += 1
            else:
                counters.replay_count += 1
            _validate_batch_review_for_snapshot(state, vector, validation)
            state, revision, replayed = _persist_batch_validation(
                store,
                state,
                revision,
                validation,
            )
            if replayed:
                counters.replay_count += 1
            continue

        if state.route == "repair_required":
            # repair_pending 会话理论上总会被 _select_investigation_targets 选中；
            # 若领域快照不满足该不变量，立即失败而不是等待或伪造中间路由。
            raise ValueError("repair_required 状态没有可调用的 repair_pending 会话")
        raise ValueError("调查批次既未形成完整五项结果，也没有可推进的会话")


__all__ = [
    "BatchReviewer",
    "InitialInvestigator",
    "InvestigationBatchOutcome",
    "InvestigationCoordinatorConflict",
    "RepairInvestigator",
    "run_investigation_and_experiment_batch",
]
