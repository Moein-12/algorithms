from collections import abc
import typing as tp

from ._utilities import Key, UNDEFINED, identity


T = tp.TypeVar('T')
T2 = tp.TypeVar('T2')

# standard definition

def merge_in_place(seq: abc.MutableSequence[T], /, start: int, middle: int, end: int, *, key: Key[T]):
    # adapted from https://www.geeksforgeeks.org/in-place-merge-sort/

    n1 = middle - start
    n2 = end - middle

    # Create temp arrays
    L = [UNDEFINED] * n1
    R = [UNDEFINED] * n2

    # Copy data to temp arrays L[] and R[]
    for i in range(n1):
        L[i] = seq[start + i]
    for j in range(n2):
        R[j] = seq[middle + j]

    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = start  # Initial index of merged subarray

    # Merge the temp arrays back
    # into arr[start..right]
    while i < n1 and j < n2:
        if key(R[j]) < key(L[i]):
            seq[k] = L[i]
            i += 1
        else:
            seq[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[],
    # if there are any
    while i < n1:
        seq[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], 
    # if there are any
    while j < n2:
        seq[k] = R[j]
        j += 1
        k += 1

def standard_merge_sort(
    seq: abc.MutableSequence[T], /, start: int, end: int, *, key: Key[T] = identity
):
    if start + 1 < end:
        middle = (start + end) >> 1  # equivalent to (start + end) // 2

        standard_merge_sort(seq, start, middle, key=key)
        standard_merge_sort(seq, middle, end, key=key)
        merge_in_place(seq, start, middle, end, key=key)

    # elif start + 1 == end:
    #     pass  # as it's already sorted


def _merged(
    seq: abc.MutableSequence[T],
    /,
    start: int,
    middle: int,
    end: int,
    *,
    key: Key[T] = identity,
) -> None:
    raise NotImplementedError

def _merge_sorted(
    seq: abc.MutableSequence[T],
    /,
    start: int,
    middle: int,
    end: int,
    *,
    key: Key[T] = identity,
) -> None:
    raise NotImplementedError

def _merged_indexes(
    seq: abc.MutableSequence[T],
    /,
    start: int,
    middle: int,
    end: int,
    *,
    key: Key[T] = identity,
) -> tuple[int, int]:
    raise NotImplementedError

def _merge_sorted_indexes(
    seq: abc.MutableSequence[T],
    /,
    start: int,
    middle: int,
    end: int,
    *,
    key: Key[T] = identity,
) -> tuple[int, int]:
    raise NotImplementedError

def _enumerate_merge_sorted(
    seq: abc.MutableSequence[T],
    /,
    start: int,
    middle: int,
    end: int,
    *,
    key: Key[T] = identity,
) -> tuple[int, int]:
    raise NotImplementedError

