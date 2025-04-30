from collections import abc
from itertools import pairwise, islice
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
    
    else:
        raise IndexError(f'Invalid range: [{start}, {end})')


def _merged(
    gens: abc.MutableSequence[abc.Iterator[T]],
    key: Key[T] = None,
) -> abc.Generator[T]:
    items = [next(gen, UNDEFINED) for gen in gens]
    active_count = len(items)
    get_key = items.__getitem__ if key is None else lambda i: key(items[i])  # type: tp.Any
    
    for i in range(active_count - 1, -1, -1):
        if items[i] is UNDEFINED:
            items[i], items[active_count - 1] = items[active_count - 1], items[i]
            gens[i], gens[active_count - 1] = gens[active_count - 1], gens[i]
            active_count -= 1
    
    while active_count > 0:
        min_index = min(range(active_count), key=get_key)
        yield items[min_index]

        next_value = next(gens[min_index], UNDEFINED)

        if next_value is UNDEFINED:
            active_count -= 1
            items[min_index], items[active_count] = items[active_count], items[min_index]
            gens[min_index], gens[active_count] = gens[active_count], gens[min_index]
        else:
            items[min_index] = next_value

def _merge_sorted(
    seq: abc.Sequence[T],
    /,
    start: int,
    end: int,
    *,
    k: int = 2,
    key: Key[T] = identity,
) -> abc.Iterator[T]:
    if end - start < 2:
        return iter(seq[start:end])

    else:
        length = end - start
        return _merged(
            [_merge_sorted(seq, low, high, k=k, key=key) for low, high in pairwise(
                start + (length * ind // k) for ind in range(k + 1)
            )], key=key,
        )
