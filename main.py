from numpy import array


def show(title: str, values) -> None:
    print("=" * 60)
    print(title)
    print(f"shape={values.shape}, dtype={values.dtype.__name__}")
    print("-" * 60)
    print(values)
    print()


def main() -> None:
    show(
        "1D — 긴 정수",
        array([10**18, 10**19, -(10**20)]),
    )

    show(
        "1D — 과학 표기법 / 혼합 float",
        array([1.23456789e-45, 123456789.123456789, 1.23456789e200]),
    )

    show(
        "1D — 복소수 (과학 표기법)",
        array([1 + 2e20j, 3e-15 + 4e15j, 5e200 + 6e-200j]),
    )

    show(
        "1D — nan / inf",
        array([float("nan"), float("inf"), float("-inf"), 1.0]),
    )

    show(
        "1D — 긴 문자열",
        array(["a", "x" * 500, "hello"]),
    )

    show(
        "1D — 80자 경계 (10**75)",
        array([10**75]),
    )

    show(
        "2D — 긴 정수 (행 라벨 / 열 정렬)",
        array([[10**15, 10**16], [10**17, 10**18]]),
    )

    show(
        "2D — 과학 표기법",
        array([[1e-30, 1e30], [1e300, 1e-300]]),
    )

    show(
        "2D — 복소수",
        array([[1 + 1j, 1e100 + 1e-100j], [1e-200, 3 + 4j]]),
    )

    show(
        "2D — 긴 문자열 (잘림)",
        array([["short", "y" * 40], ["z" * 30, "ok"]]),
    )

    show(
        "2D — 단일 거대 값 (10**50)",
        array([[10**50]]),
    )

    show(
        "2D — 단일 초장 문자열 (100자, 잘림)",
        array([["x" * 100]]),
    )

    show(
        "3D — 마지막 2 dim = 행렬 블록 (2 x 3 x 12)",
        array([10 ** (index % 8) for index in range(2 * 3 * 12)]).reshape(
            2, 3, 12
        ),
    )

    show(
        "3D — 큰 텐서, 열 생략 (2 x 4 x 20)",
        array(range(2 * 4 * 20)).reshape(2, 4, 20),
    )

    show(
        "1D — 원소 매우 많음 (5000)",
        array(range(5000)),
    )

    show(
        "2D — 행 매우 많음 (100 x 5)",
        array(range(500)).reshape(100, 5),
    )

    show(
        "2D — 열 매우 많음 (5 x 100)",
        array(range(500)).reshape(5, 100),
    )

    show(
        "3D — 블록 매우 많음 (50 x 5 x 20)",
        array(range(5000)).reshape(50, 5, 20),
    )

    show(
        "3D — 행 매우 많음 (2 x 100 x 10)",
        array(range(2000)).reshape(2, 100, 10),
    )

    show(
        "4D — 외부 차원 많음 (10 x 10 x 10 x 10)",
        array(range(10000)).reshape(10, 10, 10, 10),
    )


if __name__ == "__main__":
    main()
