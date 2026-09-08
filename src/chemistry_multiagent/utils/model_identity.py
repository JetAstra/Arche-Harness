"""共享模型身份比较规则。"""

from __future__ import annotations


def model_identity_matches(
    configured_model: str | None,
    served_model: str | None,
) -> bool:
    """判断服务返回的模型身份是否对应配置入口。

    先比较去首尾空白后的完整标识；兼容代理给命名模型返回 basename 的情况时，
    再仅比较两个标识最后一个 ``/`` 分段的大小写折叠结果。不会把任意子串当成
    模型身份匹配。
    """

    if not isinstance(configured_model, str) or not isinstance(served_model, str):
        return False
    configured = configured_model.strip()
    served = served_model.strip()
    if not configured or not served:
        return False
    if configured == served:
        return True
    return configured.rsplit("/", 1)[-1].casefold() == served.rsplit("/", 1)[-1].casefold()


__all__ = ["model_identity_matches"]
